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
# 설정 (실전용)
# ==========================
SAMPLES_PER_SHARD = 50_000_000        # 샤드당 최대 샘플 수 (5천만)
MAP_SIZE_BASE = 64 * 1024 * 1024 * 1024    # LMDB map_size 시작값 = 64GB
COMMIT_INTERVAL = 10_000
MAX_GAME_RETRIES = 3
SHARDS_PER_RUN = 3                    # 실행 1번당 최대 샤드 개수

# ==========================
# 전역 (LMDB 상태)
# ==========================
env = None
txn = None
shard_dir = ""
samples_in_shard = 0
current_map_size = MAP_SIZE_BASE

# ==========================
# 로그
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
# Checkpoint
# ==========================
def load_checkpoint(total_games: int):
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("total_games") != total_games:
            log("[WARN] checkpoint mismatch → 무시")
            return None
        idx = data.get("last_game_index")
        if isinstance(idx, int) and 0 <= idx < total_games:
            return idx
    except Exception as e:
        log(f"[WARN] checkpoint load 실패: {e}")
    return None

def save_checkpoint(last_game, total_games, global_index, shard_index):
    data = {
        "last_game_index": int(last_game),
        "total_games": int(total_games),
        "global_index": int(global_index),
        "current_shard_index": int(shard_index),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log(f"[CHECKPOINT] last_game={last_game}, total_samples={global_index}, shard_index={shard_index:04d}")
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
def write_meta(shard_path, num_samples):
    meta = {
        "num_samples": int(num_samples),
        "tensor_shape": [18, 8, 8],
        "tensor_dtype": "uint8",
        "label_type": "policy_value"
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
                # meta 없는 샤드는 불완전 → 삭제
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
    """현재 global current_map_size 사용해서 새 샤드 open"""
    global env, txn, shard_dir, samples_in_shard, current_map_size

    # 새 샤드를 열 때는 항상 초기 map_size인 64GB로 설정
    shard_dir = os.path.join(OUT_ROOT, f"shard_{shard_index:04d}")
    os.makedirs(shard_dir, exist_ok=True)

    # 새로운 샤드는 항상 초기 map_size인 64GB로 설정
    env = lmdb.open(
        shard_dir,
        map_size=MAP_SIZE_BASE,  # 초기 값 64GB로 설정
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=False,
        map_async=False,
    )
    txn = env.begin(write=True)
    samples_in_shard = 0

    log(f"[SHARD] open shard_{shard_index:04d} (map_size={MAP_SIZE_BASE})")

def finalize_current_shard():
    """현재 샤드를 마무리 (commit + meta + close)"""
    global env, txn, shard_dir, samples_in_shard

    if env is None:
        return

    try:
        if txn is not None:
            txn.commit()
            env.sync()
    except Exception as e:
        log(f"[WARN] finalize commit 실패(무시): {e}")

    write_meta(shard_dir, samples_in_shard)

    try:
        env.close()
    except Exception:
        pass

    env = None
    txn = None

# ==========================
# 인덱싱 (텍스트 기반, tell 사용)
# ==========================
def index_pgn_file():
    """
    mmap + 바이트 검색으로 [Event ] 오프셋만 뽑는 초고속 인덱싱.
    PGN은 기본적으로 ASCII/UTF-8이므로,
    텍스트 모드(open(..., newline=""))에서 f.seek(pos) 해도 위치가 일치한다.
    """
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
# 메인 실행
# ==========================
def main():
    global env, txn, shard_dir, samples_in_shard, current_map_size

    os.makedirs(OUT_ROOT, exist_ok=True)
    log("===== NEW RUN STARTED =====")

    # 기존 샤드 스캔
    existing = scan_shards()
    total_existing = sum(x[2] for x in existing)
    next_shard = max([x[0] for x in existing], default=-1) + 1

    log(f"[INFO] existing shards={len(existing)}, total_samples={total_existing}")
    log(f"[INFO] next shard index = {next_shard:04d}")

    # 인덱싱
    offsets = index_pgn_file()
    total_games = len(offsets)

    # 체크포인트
    start_idx = load_checkpoint(total_games)
    start_idx = 0 if start_idx is None else start_idx + 1
    log(f"[INFO] start from game {start_idx}/{total_games}")

    shard_index = next_shard
    current_map_size = MAP_SIZE_BASE
    open_new_shard(shard_index)

    samples_in_shard = 0
    global_index = total_existing
    shards_created = 0

    pbar = tqdm(total=total_games, initial=start_idx, desc="Writing LMDB", unit="game")

    # PGN 파일 열기 (인덱싱과 동일한 모드)
    with open(PGN_PATH, "r", encoding="utf-8", errors="replace", newline="") as f:
        game_idx = start_idx
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

                    moves = list(game.mainline_moves())
                    move_count = len(moves)

                    # 샤드 용량 (샘플 수 기준) 체크
                    if samples_in_shard + move_count > SAMPLES_PER_SHARD:
                        finalize_current_shard()
                        shards_created += 1
                        if shards_created >= SHARDS_PER_RUN:
                            pbar.close()
                            save_checkpoint(
                                last_game=game_idx,
                                total_games=total_games,
                                global_index=global_index,
                                shard_index=shard_index,
                            )
                            log(f"[DONE] total samples={global_index}")
                            log("===== RUN FINISHED =====")
                            return

                        shard_index += 1
                        open_new_shard(shard_index)

                    board = game.board()

                    for mv in moves:
                        arr = board_to_tensor(board)
                        arr = np.asarray(arr, dtype=np.uint8)
                        tensor_bytes = arr.tobytes()

                        policy = mv.from_square * 64 + mv.to_square
                        value = base_value if board.turn == chess.WHITE else -base_value

                        obj = {
                            "tensor": tensor_bytes,
                            "shape": list(arr.shape),
                            "dtype": "uint8",
                            "label": {
                                "policy": int(policy),
                                "value": float(value),
                            },
                        }

                        key = f"{global_index:012d}".encode()
                        data = msgpack.packb(obj, use_bin_type=True)

                        # MapFull 방어: map_size 초과 시 자동 샤드 롤오버 + map_size 증가
                        while True:
                            try:
                                txn.put(key, data)
                                break
                            except lmdb.MapFullError as e:
                                log(
                                    f"[WARN] MapFullError at global_index={global_index}, shard={shard_index:04d}: {e}"
                                )
                                finalize_current_shard()
                                shards_created += 1
                                if shards_created >= SHARDS_PER_RUN:
                                    pbar.close()
                                    save_checkpoint(
                                        last_game=game_idx,
                                        total_games=total_games,
                                        global_index=global_index,
                                        shard_index=shard_index,
                                    )
                                    log(f"[DONE] total samples={global_index}")
                                    log("===== RUN FINISHED =====")
                                    return
                                shard_index += 1
                                current_map_size += 64 * 1024 * 1024 * 1024  # 64GB 증가
                                open_new_shard(shard_index)

                        samples_in_shard += 1
                        global_index += 1

                        if (global_index % COMMIT_INTERVAL) == 0:
                            try:
                                txn.commit()
                                env.sync()
                            except lmdb.Error as e:
                                log(f"[WARN] periodic commit 실패(무시): {e}")
                            txn = env.begin(write=True)

                        board.push(mv)

                    success = True

                except Exception as e:
                    retries += 1
                    log(f"[ERROR] game_idx={game_idx}, retry={retries}, err={e}")

            if not success:
                log(f"[WARN] skip game_idx={game_idx} (after {MAX_GAME_RETRIES} retries)")

            game_idx += 1

    pbar.close()

    # 전체 PGN 끝까지 처리한 경우
    finalize_current_shard()

    save_checkpoint(
        last_game=total_games - 1,
        total_games=total_games,
        global_index=global_index,
        shard_index=shard_index,
    )

    log(f"[DONE] total samples={global_index}")
    log("===== RUN FINISHED =====")


if __name__ == "__main__":
    main()
