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
SAMPLES_PER_SHARD = 50_000_000               # 샤드당 최대 샘플 수
MAP_SIZE_BASE = 64 * 1024 * 1024 * 1024      # 64GB
MAX_GAME_RETRIES = 3
SHARDS_PER_RUN = 3                           # 실행 1번당 최대 생성 shard 수

# ==========================
# 전역 상태
# ==========================
env = None
txn = None
shard_dir = ""
samples_in_shard = 0
current_map_size = MAP_SIZE_BASE

# commit / resume용
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
    except Exception:
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
            log("[WARN] checkpoint mismatch(total_games) → 무시")
            return None
        idx = data.get("last_game_index")
        if isinstance(idx, int) and 0 <= idx < total_games:
            return data
    except Exception as e:
        log(f"[WARN] checkpoint load 실패: {e}")
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
        log(f"[WARN] checkpoint 저장 실패: {e}")

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
# Shard 관리
# ==========================
def write_meta(shard_path, num_samples, map_size_bytes):
    meta = {
        "num_samples": int(num_samples),
        "tensor_shape": [18, 8, 8],
        "tensor_dtype": "uint8",
        "label_type": "policy_value",
        "map_size_bytes": int(map_size_bytes),
        "map_size_gb": float(map_size_bytes / (1024**3)),
    }
    with open(os.path.join(shard_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def scan_shards():
    result = []
    if not os.path.exists(OUT_ROOT):
        return result
    for d in os.listdir(OUT_ROOT):
        if d.startswith("shard_") and len(d) == 10:
            shard_path = os.path.join(OUT_ROOT, d)
            meta_path = os.path.join(shard_path, "meta.json")
            if not os.path.exists(meta_path):
                # meta 없는 shard는 불완전 → 삭제
                shutil.rmtree(shard_path, ignore_errors=True)
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                num = int(meta.get("num_samples", 0))
            except Exception:
                num = 0
            idx = int(d.split("_")[1])
            result.append((idx, shard_path, num))
    result.sort(key=lambda x: x[0])
    return result

def open_new_shard(shard_index: int):
    """
    새 shard는 항상 map_size = 64GB로 시작.
    """
    global env, txn, shard_dir, samples_in_shard, current_map_size

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
        readahead=True,      # HDD 순차 I/O 최적
        map_async=False,     # 무결성 우선 (commit=durable 가정)
    )
    txn = env.begin(write=True)

    log(f"[SHARD] open shard_{shard_index:04d} (map_size={current_map_size // (1024**3)}GB)")

def grow_map_size_for_current_shard():
    """
    MapFull 발생 시: 같은 shard에서 map_size를 64GB 늘리고 재-open.
    현재 게임은 전부 다시 쓸 것이므로 txn.abort() 후 재시작.
    """
    global env, txn, shard_dir, current_map_size

    old_gb = current_map_size // (1024**3)
    new_map_size = current_map_size + MAP_SIZE_BASE
    new_gb = new_map_size // (1024**3)

    log(f"[MAPSIZE] MapFull → {os.path.basename(shard_dir)} {old_gb}GB → {new_gb}GB")

    try:
        if txn is not None:
            txn.abort()
    except Exception as e:
        log(f"[WARN] txn.abort 실패(무시): {e}")

    try:
        if env is not None:
            env.close()
    except Exception:
        pass

    current_map_size = new_map_size

    env = lmdb.open(
        shard_dir,
        map_size=current_map_size,
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=True,
        map_async=False,
    )
    txn = env.begin(write=True)

    log(f"[MAPSIZE] 재-open 완료 (map_size={current_map_size // (1024**3)}GB)")

def finalize_current_shard():
    """
    현재 shard 마무리 (commit + sync + meta.json + close)
    """
    global env, txn, shard_dir, samples_in_shard, current_map_size

    if env is None:
        return

    try:
        if txn is not None:
            txn.commit()
        try:
            env.sync()
        except lmdb.Error as e:
            log(f"[WARN] env.sync 실패(무시): {e}")
    except Exception as e:
        log(f"[WARN] finalize commit 실패(무시): {e}")

    write_meta(shard_dir, samples_in_shard, current_map_size)

    try:
        env.close()
    except Exception:
        pass

    env = None
    txn = None

# ==========================
# 인덱싱 (mmap)
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
# 게임 → 샘플 버퍼
# ==========================
def build_game_samples(game, base_value, global_index_start):
    """
    한 게임의 모든 수 → [(key_int, tensor_bytes, shape, policy, value), ...]
    """
    board = game.board()
    samples = []
    key_int = global_index_start

    for mv in game.mainline_moves():
        arr = board_to_tensor(board)
        arr = np.asarray(arr, dtype=np.uint8)
        tensor_bytes = arr.tobytes()

        policy = mv.from_square * 64 + mv.to_square
        value = base_value if board.turn == chess.WHITE else -base_value

        samples.append(
            (key_int, tensor_bytes, arr.shape, policy, float(value))
        )

        key_int += 1
        board.push(mv)

    return samples

def write_game_samples(game_samples):
    """
    한 게임 단위로 LMDB에 기록.
    MapFull 나오면 같은 shard에서 map_size를 키우고
    게임 전체를 처음부터 다시 put.
    """
    global txn

    while True:
        try:
            for key_int, tensor_bytes, shape, policy, value in game_samples:
                key = f"{key_int:012d}".encode()
                obj = {
                    "tensor": tensor_bytes,
                    "shape": list(shape),
                    "dtype": "uint8",
                    "label": {
                        "policy": int(policy),
                        "value": float(value),
                    },
                }
                data = msgpack.packb(obj, use_bin_type=True)
                txn.put(key, data)
            break
        except lmdb.MapFullError as e:
            log(f"[WARN] MapFullError during game write (first key={game_samples[0][0]}): {e}")
            grow_map_size_for_current_shard()
            # txn 새로 열렸으니 같은 game_samples 전체를 다시 시도

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
    next_shard = max([x[0] for x in existing], default=-1) + 1

    log(f"[INFO] existing shards={len(existing)}, total_samples={total_existing}")
    log(f"[INFO] next shard index = {next_shard:04d}")

    offsets = index_pgn_file()
    total_games = len(offsets)

    # checkpoint 로드
    cp = load_checkpoint(total_games)
    if cp is None:
        start_game_idx = 0
        global_index = total_existing
    else:
        start_game_idx = cp["last_game_index"] + 1
        # global_index는 checkpoint 기준으로 가져감
        global_index = cp.get("global_index", total_existing)
    log(f"[INFO] start from game {start_game_idx}/{total_games}, global_index={global_index}")

    shard_index = next_shard
    open_new_shard(shard_index)

    samples_in_shard = 0
    shards_created = 0

    pbar = tqdm(total=total_games, initial=start_game_idx, desc="Writing LMDB", unit="game")

    with open(PGN_PATH, "r", encoding="utf-8", errors="replace", newline="") as f:
        game_idx = start_game_idx

        while game_idx < total_games:
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

                    # 1) 게임 샘플 버퍼 생성
                    game_samples = build_game_samples(game, base_value, global_index_start=global_index)
                    num_samples_game = len(game_samples)
                    if num_samples_game == 0:
                        success = True
                        break

                    # 2) shard 용량 체크 (게임 시작 전에만 shard 롤오버)
                    if samples_in_shard + num_samples_game > SAMPLES_PER_SHARD:
                        finalize_current_shard()
                        shards_created += 1
                        if shards_created >= SHARDS_PER_RUN:
                            pbar.close()
                            save_checkpoint(
                                last_game_index=game_idx - 1,
                                total_games=total_games,
                                global_index=global_index,
                                shard_index=shard_index,
                            )
                            log(f"[DONE] total samples={global_index}")
                            log("===== RUN FINISHED (SHARD LIMIT) =====")
                            return

                        shard_index += 1
                        open_new_shard(shard_index)
                        samples_in_shard = 0

                    # 3) 게임 샘플 기록 (MapFull 시 game 단위 재시도)
                    write_game_samples(game_samples)

                    # 4) 게임 커밋 (원자성 보장)
                    try:
                        txn.commit()
                        txn = env.begin(write=True)
                    except lmdb.Error as e:
                        log(f"[ERROR] commit 실패: {e}")
                        raise

                    # 5) 게임 성공: 카운트/체크포인트 갱신
                    global_index += num_samples_game
                    samples_in_shard += num_samples_game

                    save_checkpoint(
                        last_game_index=game_idx,
                        total_games=total_games,
                        global_index=global_index,
                        shard_index=shard_index,
                    )

                    success = True

                except Exception as e:
                    retries += 1
                    log(f"[ERROR] game_idx={game_idx}, retry={retries}, err={e}")

            if not success:
                log(f"[WARN] skip game_idx={game_idx} (after {MAX_GAME_RETRIES} retries)")

            game_idx += 1

    pbar.close()

    # 전체 PGN 처리 완료
    finalize_current_shard()

    save_checkpoint(
        last_game_index=total_games - 1,
        total_games=total_games,
        global_index=global_index,
        shard_index=shard_index,
    )

    log(f"[DONE] total samples={global_index}")
    log("===== RUN FINISHED (EOF) =====")


if __name__ == "__main__":
    main()
