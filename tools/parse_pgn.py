import os
import json
import shutil
import lmdb
import msgpack
import mmap
import numpy as np
import chess
import chess.pgn
from datetime import datetime
from tqdm import tqdm
from fen_to_tensor import board_to_tensor

# ==========================
# 경로 설정
# ==========================
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))

PGN_PATH = os.path.join(PROJECT_ROOT, "data", "pgn_raw", "lichess_db_standard_rated_2025-01.pgn")
OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "lmdb", "standard_2025_01")

CHECKPOINT_PATH = os.path.join(OUT_ROOT, "checkpoint.json")
LOG_PATH = os.path.join(OUT_ROOT, "parse_pgn.log")

# ==========================
# 설정
# ==========================
SAMPLES_PER_SHARD = 50_000_000
MAP_SIZE_BASE = 64 * 1024 * 1024 * 1024    # 64GB
MAX_GAME_RETRIES = 3
SHARDS_PER_RUN = 3

# 성능 병목 제거 핵심 설정
CHECKPOINT_EVERY = 100000                 # ★ 100000게임마다 checkpoint 저장
COMMIT_INTERVAL = 100000                # ★ 성능 강화를 위한 commit 배치
RECORD_SYNC = False                     # ★ shard 종료 시에만 sync()

# ==========================
# 전역 상태
# ==========================
env = None
txn = None
shard_dir = ""
samples_in_shard = 0
current_map_size = MAP_SIZE_BASE
global_index = 0


# ==========================
# 유틸
# ==========================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(OUT_ROOT, exist_ok=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


# ==========================
# checkpoint
# ==========================
def load_checkpoint(total_games: int):
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("total_games") != total_games:
            log("[WARN] checkpoint mismatch(total_games) → ignore")
            return None
        i = data.get("last_game_index")
        if isinstance(i, int) and 0 <= i < total_games:
            return data
    except:
        pass
    return None


def save_checkpoint(last_game_index, total_games, global_index, shard_index):
    data = {
        "last_game_index": int(last_game_index),
        "total_games": int(total_games),
        "global_index": int(global_index),
        "current_shard_index": int(shard_index),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log(f"[CHECKPOINT] last_game={last_game_index}, total_samples={global_index}, shard_index={shard_index:04d}")
    except Exception as e:
        log(f"[WARN] checkpoint save failed: {e}")


# ==========================
# 결과값 처리
# ==========================
def result_to_value(r: str):
    if r == "1-0":
        return 1.0
    if r == "0-1":
        return -1.0
    if r == "1/2-1/2":
        return 0.0
    return None


# ==========================
# shard 관리
# ==========================
def write_meta(shard_path, num_samples, map_size_bytes):
    meta = {
        "num_samples": int(num_samples),
        "tensor_shape": [18, 8, 8],
        "tensor_dtype": "uint8",
        "label_type": "policy_value",
        "map_size_bytes": int(map_size_bytes),
        "map_size_gb": float(map_size_bytes / (1024 ** 3)),
    }
    with open(os.path.join(shard_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def scan_shards():
    if not os.path.exists(OUT_ROOT):
        return []
    result = []
    for d in os.listdir(OUT_ROOT):
        if d.startswith("shard_") and len(d) == 10:
            shard_path = os.path.join(OUT_ROOT, d)
            meta_path = os.path.join(shard_path, "meta.json")
            if not os.path.exists(meta_path):
                shutil.rmtree(shard_path, ignore_errors=True)
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                num = int(meta.get("num_samples", 0))
            except:
                num = 0
            idx = int(d.split("_")[1])
            result.append((idx, shard_path, num))
    result.sort(key=lambda x: x[0])
    return result


def open_new_shard(shard_index: int):
    global env, txn, shard_dir, samples_in_shard, current_map_size

    shard_dir = os.path.join(OUT_ROOT, f"shard_{shard_index:04d}")
    os.makedirs(shard_dir, exist_ok=True)

    samples_in_shard = 0
    current_map_size = MAP_SIZE_BASE

    env = lmdb.open(
        shard_dir,
        map_size=current_map_size,
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=True,
        map_async=True,
    )
    txn = env.begin(write=True)

    log(f"[SHARD] open shard_{shard_index:04d} (map_size=64GB)")


def grow_map_size_for_current_shard():
    global env, txn, shard_dir, current_map_size

    old_gb = current_map_size // (1024 ** 3)
    current_map_size += MAP_SIZE_BASE
    new_gb = current_map_size // (1024 ** 3)

    log(f"[MAPSIZE] MapFull → {old_gb}GB → {new_gb}GB")

    try:
        txn.abort()
    except:
        pass
    try:
        env.close()
    except:
        pass

    env = lmdb.open(
        shard_dir,
        map_size=current_map_size,
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=True,
        map_async=True,
    )
    txn = env.begin(write=True)

    log(f"[MAPSIZE] reopen done (map_size={new_gb}GB)")


def finalize_current_shard():
    global env, txn, shard_dir, samples_in_shard, current_map_size

    if env is None:
        return

    try:
        txn.commit()
        if RECORD_SYNC:
            env.sync()
    except:
        pass

    write_meta(shard_dir, samples_in_shard, current_map_size)

    try:
        env.close()
    except:
        pass

    env = None
    txn = None


# ==========================
# 인덱싱
# ==========================
def index_pgn_file():
    log("[INFO] fast indexing (mmap, [Event ] offsets)...")

    offsets = []
    pattern = b"[Event "
    size = os.path.getsize(PGN_PATH)

    with open(PGN_PATH, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pbar = tqdm(total=size, unit="B", unit_scale=True, desc="Indexing")
        last = 0

        pos = mm.find(pattern)
        while pos != -1:
            offsets.append(pos)
            pbar.update(pos - last)
            last = pos
            pos = mm.find(pattern, pos + 1)

        pbar.update(size - last)
        pbar.close()
        mm.close()

    log(f"[INFO] indexing complete: {len(offsets)} games")
    return offsets


# ==========================
# 게임 → 버퍼
# ==========================
def build_game_samples(game, base_value, global_index_start):
    board = game.board()
    samples = []
    key_int = global_index_start

    for mv in game.mainline_moves():
        arr = board_to_tensor(board)
        arr = np.asarray(arr, dtype=np.uint8)
        tensor_bytes = arr.tobytes()

        policy = mv.from_square * 64 + mv.to_square
        value = base_value if board.turn == chess.WHITE else -base_value

        samples.append((key_int, tensor_bytes, arr.shape, policy, float(value)))

        key_int += 1
        board.push(mv)

    return samples


def write_game_samples(game_samples):
    global txn

    while True:
        try:
            for key_int, tensor_bytes, shape, policy, value in game_samples:
                key = f"{key_int:012d}".encode()
                obj = {
                    "tensor": tensor_bytes,
                    "shape": list(shape),
                    "dtype": "uint8",
                    "label": {"policy": policy, "value": value},
                }
                data = msgpack.packb(obj, use_bin_type=True)
                txn.put(key, data)
            break
        except lmdb.MapFullError:
            grow_map_size_for_current_shard()


# ==========================
# 메인
# ==========================
def main():
    global env, txn, shard_dir, samples_in_shard, current_map_size, global_index

    os.makedirs(OUT_ROOT, exist_ok=True)
    log("===== NEW RUN STARTED =====")

    # 기존 shard 스캔
    existing = scan_shards()
    total_existing = sum(x[2] for x in existing)
    shard_index = max([x[0] for x in existing], default=-1) + 1

    log(f"[INFO] existing shards={len(existing)}, total_samples={total_existing}")
    log(f"[INFO] next shard index = {shard_index:04d}")

    offsets = index_pgn_file()
    total_games = len(offsets)

    # checkpoint 로드
    cp = load_checkpoint(total_games)
    if cp:
        start_game_idx = cp["last_game_index"] + 1
        global_index = cp["global_index"]
        shard_index = cp["current_shard_index"]
    else:
        start_game_idx = 0
        global_index = total_existing

    log(f"[INFO] start from game {start_game_idx}/{total_games}, global_index={global_index}")

    open_new_shard(shard_index)

    pbar = tqdm(total=total_games, initial=start_game_idx, desc="Writing LMDB", unit="game")

    with open(PGN_PATH, "r", encoding="utf-8", errors="replace") as f:
        for game_idx in range(start_game_idx, total_games):

            pbar.update(1)
            pos = offsets[game_idx]

            retries = 0
            success = False

            while retries < MAX_GAME_RETRIES and not success:
                try:
                    f.seek(pos)
                    game = chess.pgn.read_game(f)
                    if game is None:
                        success = True
                        break

                    base_value = result_to_value(game.headers.get("Result", ""))
                    if base_value is None:
                        success = True
                        break

                    game_samples = build_game_samples(game, base_value, global_index)
                    num_samples = len(game_samples)
                    if num_samples == 0:
                        success = True
                        break

                    # shard rollover
                    if samples_in_shard + num_samples > SAMPLES_PER_SHARD:
                        finalize_current_shard()
                        shard_index += 1
                        open_new_shard(shard_index)
                        samples_in_shard = 0
                        save_checkpoint(game_idx - 1, total_games, global_index, shard_index - 1)

                    # write
                    write_game_samples(game_samples)

                    # commit
                    txn.commit()
                    txn = env.begin(write=True)

                    # update counters
                    global_index += num_samples
                    samples_in_shard += num_samples

                    # checkpoint (every N games)
                    if game_idx % CHECKPOINT_EVERY == 0:
                        save_checkpoint(game_idx, total_games, global_index, shard_index)

                    success = True

                except Exception as e:
                    retries += 1
                    log(f"[ERROR] game_idx={game_idx}, retry={retries}, err={e}")

            if not success:
                log(f"[WARN] skip game_idx={game_idx} (after retries)")

    # finish
    finalize_current_shard()
    save_checkpoint(total_games - 1, total_games, global_index, shard_index)

    log(f"[DONE] total samples={global_index}")
    log("===== RUN FINISHED (EOF) =====")


if __name__ == "__main__":
    main()
