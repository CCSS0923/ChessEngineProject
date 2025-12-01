import os
import json
import shutil
import mmap
import lmdb
import msgpack
import numpy as np
import chess
import chess.pgn
from datetime import datetime
from tqdm import tqdm
from fen_to_tensor import board_to_tensor


# =========================================
# 경로
# =========================================
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))

PGN_PATH = os.path.join(PROJECT_ROOT, "data", "pgn_raw", "lichess_db_standard_rated_2025-01.pgn")
OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "lmdb", "standard_2025_01")

CHECKPOINT_PATH = os.path.join(OUT_ROOT, "checkpoint.json")
LOG_PATH = os.path.join(OUT_ROOT, "parse_pgn.log")


# =========================================
# 설정
# =========================================
SAMPLES_PER_SHARD = 50_000_000

MAP_SIZE_BASE = 64 * 1024 * 1024 * 1024      # 64GB 시작
MAP_SIZE_STEP = 64 * 1024 * 1024 * 1024      # MapFull 시 64GB 증가

MAX_GAME_RETRIES = 3
SHARDS_PER_RUN = 3

COMMIT_INTERVAL = 100_000
CHECKPOINT_EVERY = 100_000

RECORD_SYNC = False  # shard 종료 시에만 sync()


# =========================================
# 전역 변수
# =========================================
env = None
txn = None
shard_dir = ""
current_map_size = MAP_SIZE_BASE
samples_in_shard = 0
global_index = 0
last_commit_index = 0


# =========================================
# 로그
# =========================================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)

    os.makedirs(OUT_ROOT, exist_ok=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


# =========================================
# Checkpoint
# =========================================
def load_checkpoint(total_games: int):
    if not os.path.exists(CHECKPOINT_PATH):
        return None

    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            js = json.load(f)

        if js.get("total_games") != total_games:
            log("[WARN] checkpoint mismatch → ignore")
            return None

        return js
    except:
        return None


def save_checkpoint(last_game_idx, total_games, global_idx, shard_idx):
    data = {
        "last_game_index": int(last_game_idx),
        "total_games": int(total_games),
        "global_index": int(global_idx),
        "current_shard_index": int(shard_idx),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        log(f"[CHECKPOINT] last_game={last_game_idx}, total={global_idx}, shard={shard_idx:04d}")
    except Exception as e:
        log(f"[WARN] checkpoint save failed: {e}")


# =========================================
# 결과 변환
# =========================================
def result_to_value(r: str):
    if r == "1-0":
        return 1.0
    if r == "0-1":
        return -1.0
    if r == "1/2-1/2":
        return 0.0
    return None


# =========================================
# Shard 관리
# =========================================
def write_meta(shard_path, num_samples, map_size_bytes):
    meta = {
        "num_samples": int(num_samples),
        "tensor_shape": [18, 8, 8],
        "tensor_dtype": "float32",
        "label_type": "policy_value",
        "map_size_bytes": int(map_size_bytes),
        "map_size_gb": float(map_size_bytes) / (1024 ** 3),
    }
    with open(os.path.join(shard_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def scan_shards():
    """불완성 shard 자동 삭제 후, 정상 shard 목록 반환"""
    if not os.path.exists(OUT_ROOT):
        return []

    result = []

    for d in os.listdir(OUT_ROOT):
        if d.startswith("shard_") and len(d) == 10:
            p = os.path.join(OUT_ROOT, d)
            m = os.path.join(p, "meta.json")

            if not os.path.exists(m):
                shutil.rmtree(p, ignore_errors=True)
                continue

            try:
                with open(m, "r", encoding="utf-8") as f:
                    js = json.load(f)
                num = int(js.get("num_samples", 0))
            except:
                num = 0

            idx = int(d.split("_")[1])
            result.append((idx, p, num))

    result.sort(key=lambda x: x[0])
    return result


def open_new_shard(shard_index: int):
    global env, txn, shard_dir, current_map_size, samples_in_shard

    shard_dir = os.path.join(OUT_ROOT, f"shard_{shard_index:04d}")
    os.makedirs(shard_dir, exist_ok=True)

    current_map_size = MAP_SIZE_BASE
    samples_in_shard = 0

    env = lmdb.open(
        shard_dir,
        map_size=current_map_size,
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=False,
        map_async=False,
    )

    txn = env.begin(write=True)

    log(f"[SHARD] open shard_{shard_index:04d} (map_size=64GB)")


def grow_map_size_for_current_shard():
    global env, txn, shard_dir, current_map_size

    old = current_map_size
    current_map_size += MAP_SIZE_STEP

    log(f"[MAPSIZE] MapFull → {old // (1024**3)}GB → {current_map_size // (1024**3)}GB")

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
        readahead=False,
        map_async=False,
    )

    txn = env.begin(write=True)


def finalize_current_shard():
    global env, txn, current_map_size, samples_in_shard, shard_dir

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

    log(f"[SHARD] finalized {os.path.basename(shard_dir)} samples={samples_in_shard}")

    env = None
    txn = None


# =========================================
# 인덱싱
# =========================================
def index_pgn_file():
    log("[INFO] indexing PGN...")
    offsets = []

    size = os.path.getsize(PGN_PATH)
    pattern = b"[Event "

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


# =========================================
# 샘플 생성
# =========================================
def build_samples_for_game(game, base_value, start_idx):
    board = game.board()
    samples = []
    idx = start_idx

    for mv in game.mainline_moves():
        arr = board_to_tensor(board)
        arr = np.asarray(arr, dtype=np.float32)

        if arr.shape != (18, 8, 8):
            raise ValueError(f"Unexpected tensor shape: {arr.shape}")

        tb = arr.tobytes()
        policy = mv.from_square * 64 + mv.to_square
        value = base_value if board.turn == chess.WHITE else -base_value

        samples.append((idx, tb, policy, float(value)))

        idx += 1
        board.push(mv)

    return samples


# =========================================
# LMDB 쓰기
# =========================================
def write_samples(samples):
    global txn

    while True:
        try:
            for idx, tb, policy, value in samples:
                key = f"{idx:012d}".encode()
                obj = {
                    "tensor": tb,
                    "shape": [18, 8, 8],
                    "dtype": "float32",
                    "label": {"policy": policy, "value": value},
                }
                data = msgpack.packb(obj, use_bin_type=True)
                txn.put(key, data)
            break
        except lmdb.MapFullError:
            grow_map_size_for_current_shard()


# =========================================
# 메인
# =========================================
def main():
    global env, txn, current_map_size
    global global_index, samples_in_shard, last_commit_index

    os.makedirs(OUT_ROOT, exist_ok=True)
    log("===== NEW RUN STARTED =====")

    # -----------------------------------
    # LMDB self-check
    # -----------------------------------
    try:
        tdir = os.path.join(OUT_ROOT, "_lmdb_test")
        os.makedirs(tdir, exist_ok=True)
        t_env = lmdb.open(tdir, map_size=1 << 20, subdir=True)
        t = t_env.begin(write=True)
        t.put(b"x", b"y")
        t.commit()
        t_env.close()
        shutil.rmtree(tdir, ignore_errors=True)
        log("[SELF-CHECK] LMDB OK")
    except Exception as e:
        log(f"[SELF-CHECK] FAILED: {e}")
        return

    # -----------------------------------
    # 기존 샤드 스캔
    # -----------------------------------
    existing = scan_shards()
    total_existing = sum(x[2] for x in existing)
    next_shard = max([x[0] for x in existing], default=-1) + 1

    log(f"[INFO] existing_shards={len(existing)}, total_samples={total_existing}")
    log(f"[INFO] next_shard={next_shard:04d}")

    # -----------------------------------
    # 인덱싱
    # -----------------------------------
    offsets = index_pgn_file()
    total_games = len(offsets)

    # -----------------------------------
    # checkpoint 복구
    # -----------------------------------
    cp = load_checkpoint(total_games)
    if cp:
        start_game = cp["last_game_index"] + 1
        global_index = cp["global_index"]
        shard_index = next_shard   # 새 샤드부터 시작
    else:
        start_game = 0
        global_index = total_existing
        shard_index = next_shard

    last_commit_index = global_index

    log(f"[INFO] start_game={start_game}, global_index={global_index}, shard_index={shard_index:04d}")

    # -----------------------------------
    # 첫 샤드 오픈
    # -----------------------------------
    open_new_shard(shard_index)

    # -----------------------------------
    # 메인 루프
    # -----------------------------------
    pbar = tqdm(total=total_games, initial=start_game, desc="Writing LMDB", unit="game")

    with open(PGN_PATH, "r", encoding="utf-8", errors="replace", newline="") as f:
        for game_idx in range(start_game, total_games):
            pbar.update(1)

            pos = offsets[game_idx]
            retries = 0
            ok = False

            while retries < MAX_GAME_RETRIES and not ok:
                try:
                    f.seek(pos)
                    game = chess.pgn.read_game(f)
                    if game is None:
                        ok = True
                        break

                    base_value = result_to_value(game.headers.get("Result", ""))
                    if base_value is None:
                        ok = True
                        break

                    samples = build_samples_for_game(game, base_value, global_index)
                    c = len(samples)
                    if c == 0:
                        ok = True
                        break

                    # -----------------------------------
                    # 샤드 용량 초과 → rollover 먼저 처리
                    # -----------------------------------
                    if samples_in_shard + c > SAMPLES_PER_SHARD:
                        finalize_current_shard()
                        shard_index += 1
                        open_new_shard(shard_index)
                        samples_in_shard = 0

                    # -----------------------------------
                    # 샘플 기록
                    # -----------------------------------
                    write_samples(samples)

                    # -----------------------------------
                    # 카운터 갱신
                    # -----------------------------------
                    global_index += c
                    samples_in_shard += c

                    # -----------------------------------
                    # COMMIT_INTERVAL 기반 배치 커밋
                    # -----------------------------------
                    if (global_index - last_commit_index) >= COMMIT_INTERVAL:
                        try:
                            txn.commit()
                            txn = env.begin(write=True)
                        except:
                            txn = env.begin(write=True)
                        last_commit_index = global_index

                    ok = True

                except Exception as e:
                    retries += 1
                    log(f"[ERROR] game_idx={game_idx}, retry={retries}, err={e}")

            if not ok:
                log(f"[WARN] SKIP game {game_idx}")

            # -----------------------------------
            # 샤드 생성 제한: “샘플 기록 후” 체크
            # -----------------------------------
            if (shard_index - next_shard + 1) >= SHARDS_PER_RUN:
                finalize_current_shard()
                save_checkpoint(game_idx, total_games, global_index, shard_index)
                log("[RUN-END] SHARDS_PER_RUN limit reached")
                return

            # -----------------------------------
            # checkpoint 주기
            # -----------------------------------
            if game_idx % CHECKPOINT_EVERY == 0:
                save_checkpoint(game_idx, total_games, global_index, shard_index)

    # -----------------------------------
    # PGN 끝까지 처리
    # -----------------------------------
    finalize_current_shard()
    save_checkpoint(total_games - 1, total_games, global_index, shard_index)

    log(f"[DONE] total_samples={global_index}")
    log("===== RUN FINISHED =====")


if __name__ == "__main__":
    main()
