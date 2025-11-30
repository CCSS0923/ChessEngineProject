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
MAP_SIZE = 64 * 1024 * 1024 * 1024    # LMDB map_size = 64GB
COMMIT_INTERVAL = 10_000
MAX_GAME_RETRIES = 3
SHARDS_PER_RUN = 3                    # 실행 1번당 최대 샤드 개수

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
    except:
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
def open_shard(shard_index, map_size):
    """map_size를 증가시키며 샤드를 열고 데이터 계속 추가"""
    shard_dir = os.path.join(OUT_ROOT, f"shard_{shard_index:04d}")
    os.makedirs(shard_dir, exist_ok=True)

    env = lmdb.open(
        shard_dir,
        map_size=map_size,
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=False,
        map_async=False,
    )
    log(f"[SHARD] open shard_{shard_index:04d} with map_size={map_size}")
    return env, shard_dir


def write_meta(shard_dir, num_samples):
    meta = {
        "num_samples": int(num_samples),
        "tensor_shape": [18, 8, 8],
        "tensor_dtype": "uint8",
        "label_type": "policy_value"
    }
    with open(os.path.join(shard_dir, "meta.json"), "w", encoding="utf-8") as f:
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
            except:
                num = 0
            idx = int(d.split("_")[1])
            result.append((idx, shard_path, num))
    result.sort(key=lambda x: x[0])
    return result

# ==========================
# 인덱싱 (캐시 없이 항상 새로)
# ==========================
def index_pgn_file():
    log("[INFO] fast indexing (Event offsets)...")
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
    map_size = MAP_SIZE  # 초기 map_size 설정
    env, shard_dir = open_shard(shard_index, map_size)  # 첫 샤드 열기
    txn = env.begin(write=True)
    samples_in_shard = 0
    global_index = total_existing
    shards_created = 0

    pbar = tqdm(total=total_games, initial=start_idx, desc="Writing LMDB", unit="game")

    game_idx = start_idx
    while game_idx < total_games:
        pbar.update(1)

        pos = offsets[game_idx]
        retries = 0
        success = False

        while retries < MAX_GAME_RETRIES and not success:
            try:
                f.seek(pos)  # 이 부분에서 'f'를 파일 핸들러로 제대로 정의
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

                # 샤드 용량 초과 시, 현재 샤드에서 map_size 증가 후 계속 처리
                projected = samples_in_shard + move_count
                while projected > SAMPLES_PER_SHARD:
                    # 현재 샤드에서 map_size 확장
                    map_size += (64 * 1024 * 1024 * 1024)  # 64GB씩 증가
                    finalize_current_shard()  # 현재 샤드 저장
                    open_new_shard(shard_index, map_size)  # 증가된 map_size로 새 샤드 열기
                    projected -= SAMPLES_PER_SHARD  # 넘친 만큼 계속 처리
                    samples_in_shard = 0  # 샤드 리셋

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
                            "value": float(value)
                        },
                    }

                    key = f"{global_index:012d}".encode()
                    data = msgpack.packb(obj, use_bin_type=True)

                    # MapFull 방어: map_size 초과 시 자동 샤드 롤오버
                    while True:
                        try:
                            txn.put(key, data)
                            break
                        except lmdb.MapFullError as e:
                            log(f"[WARN] MapFullError at global_index={global_index}, shard={shard_index:04d}: {e}")
                            finalize_current_shard()
                            shards_created += 1
                            if shards_created >= SHARDS_PER_RUN:
                                pbar.close()
                                save_checkpoint(
                                    last_game=game_idx,
                                    total_games=total_games,
                                    global_index=global_index,
                                    shard_index=shard_index
                                )
                                log(f"[DONE] total samples={global_index}")
                                log("===== RUN FINISHED =====")
                                return
                            shard_index += 1
                            map_size = map_size + (64 * 1024 * 1024 * 1024)  # 64GB씩 증가
                            open_new_shard(shard_index, map_size)

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
    if samples_in_shard > 0 and env is not None and txn is not None:
        finalize_current_shard()

    save_checkpoint(
        last_game=total_games - 1,
        total_games=total_games,
        global_index=global_index,
        shard_index=shard_index
    )

    log(f"[DONE] total samples={global_index}")
    log("===== RUN FINISHED =====")


if __name__ == "__main__":
    main()
