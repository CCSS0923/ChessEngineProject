import os
import json
import shutil
import mmap
from datetime import datetime
from typing import Optional, Tuple, List

import lmdb
import msgpack
import numpy as np
import chess
import chess.pgn
from tqdm import tqdm

from fen_to_tensor import board_to_tensor


TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))

PGN_PATH = os.path.join(PROJECT_ROOT, "data", "pgn_raw", "lichess_db_standard_rated_2025-01.pgn")
OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "lmdb", "standard_2025_01")

CHECKPOINT_PATH = os.path.join(OUT_ROOT, "checkpoint.json")
LOG_PATH = os.path.join(OUT_ROOT, "parse_pgn.log")

SAMPLES_PER_SHARD = 1_000_000
COMMIT_INTERVAL = 10_000
SHARDS_PER_RUN = 3
SHARD_LIMIT = 64 * 1024 * 1024 * 1024

TENSOR_SHAPE = [18, 8, 8]
TENSOR_DTYPE = "uint8"
MAX_GAME_RETRIES = 3


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


def load_checkpoint(total_games: int):
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("total_games") != total_games:
            log("[WARN] checkpoint total_games mismatch. ignore.")
            return None
        last_game = data.get("last_game_index")
        if isinstance(last_game, int) and 0 <= last_game < total_games:
            return last_game
    except Exception as e:
        log(f"[WARN] load checkpoint failed: {e}")
    return None


def save_checkpoint(last_game: int, total_games: int, global_index: int, shard_index: int):
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
        log(f"[CHECKPOINT] last_game={last_game}, total_samples={global_index}, shard_index={shard_index}")
    except Exception as e:
        log(f"[WARN] save checkpoint failed: {e}")


def result_to_base_value(r: str):
    if r == "1-0": return 1.0
    if r == "0-1": return -1.0
    if r == "1/2-1/2": return 0.0
    return None


def open_env(shard_index: int):
    shard_name = f"shard_{shard_index:04d}"
    shard_dir = os.path.join(OUT_ROOT, shard_name)
    os.makedirs(shard_dir, exist_ok=True)
    env = lmdb.open(
        shard_dir,
        map_size=1 << 40,
        subdir=True,
        readonly=False,
        lock=True,
        meminit=False,
        readahead=False,
        map_async=False,
    )
    return env, shard_dir


def get_shard_size(shard_dir: str):
    p = os.path.join(shard_dir, "data.mdb")
    return os.path.getsize(p) if os.path.exists(p) else 0


def write_meta(shard_dir, num_samples: int):
    if num_samples <= 0:
        return
    meta = {
        "num_samples": int(num_samples),
        "tensor_shape": TENSOR_SHAPE,
        "tensor_dtype": TENSOR_DTYPE,
        "label_type": "policy_value",
    }
    with open(os.path.join(shard_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def scan_and_clean_shards():
    complete = []
    if not os.path.exists(OUT_ROOT):
        return complete
    for d in os.listdir(OUT_ROOT):
        if not (d.startswith("shard_") and len(d) == 10):
            continue
        shard_dir = os.path.join(OUT_ROOT, d)
        meta_path = os.path.join(shard_dir, "meta.json")
        if not os.path.exists(meta_path):
            shutil.rmtree(shard_dir, ignore_errors=True)
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            ns = int(meta.get("num_samples", 0))
        except:
            ns = 0
        idx = int(d.split("_")[1])
        complete.append((idx, shard_dir, ns))
    complete.sort(key=lambda x: x[0])
    return complete


def index_pgn(path: str):
    log("[INFO] fast indexing (Event offsets)...")
    offsets = []
    pattern = b"[Event "

    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="Indexing")

        last = 0
        pos = mm.find(pattern)
        while pos != -1:
            offsets.append(pos)
            pbar.update(pos - last)
            last = pos
            pos = mm.find(pattern, pos + 1)

        pbar.update(file_size - last)
        pbar.close()
        mm.close()

    log(f"[INFO] indexing complete: {len(offsets)} games")
    return offsets


def main():
    try:
        os.makedirs(OUT_ROOT, exist_ok=True)
        log("===== NEW RUN STARTED =====")

        if not os.path.exists(PGN_PATH):
            log(f"[ERROR] PGN not found: {PGN_PATH}")
            return

        existing = scan_and_clean_shards()
        total_existing_samples = sum(x[2] for x in existing)
        next_shard_index = max([x[0] for x in existing], default=-1) + 1

        log(f"[INFO] existing shards={len(existing)}, total_samples={total_existing_samples}")
        log(f"[INFO] next shard index = {next_shard_index:04d}")

        offsets = index_pgn(PGN_PATH)
        total_games = len(offsets)
        if total_games == 0:
            log("[ERROR] No games found.")
            return

        last_game = load_checkpoint(total_games)
        start_idx = (last_game + 1) if last_game is not None else 0
        if start_idx >= total_games:
            log("[INFO] all games processed.")
            return

        log(f"[INFO] start from game {start_idx}/{total_games}")

        env, shard_dir = open_env(next_shard_index)
        txn = env.begin(write=True)

        shard_index = next_shard_index
        samples_in_shard = 0
        global_index = total_existing_samples
        shards_created = 0
        stop_run = False

        def finalize_shard():
            nonlocal env, txn, shard_dir, samples_in_shard, total_existing_samples
            txn.commit()
            env.sync()
            env.close()
            write_meta(shard_dir, samples_in_shard)
            total_existing_samples += samples_in_shard
            log(f"[SHARD] finalized shard_{shard_index:04d} samples={samples_in_shard}")
            samples_in_shard = 0

        with open(PGN_PATH, "r", encoding="utf-8", errors="replace") as f:
            pbar = tqdm(total=total_games, initial=start_idx, desc="Writing LMDB", unit="game")

            for game_idx in range(start_idx, total_games):
                if stop_run:
                    break

                pbar.update(1)
                pos = offsets[game_idx]

                retries = 0
                success = False

                while retries < MAX_GAME_RETRIES and not success:
                    try:
                        f.seek(pos)
                        game = chess.pgn.read_game(f)
                        if game is None:
                            log(f"[WARN] game {game_idx} read None → skip")
                            success = True
                            break

                        base_value = result_to_base_value(game.headers.get("Result", ""))
                        if base_value is None:
                            success = True
                            break

                        board = game.board()
                        moves = list(game.mainline_moves())

                        projected_samples = samples_in_shard + len(moves)
                        projected_size = get_shard_size(shard_dir)

                        if projected_samples >= SAMPLES_PER_SHARD or projected_size >= SHARD_LIMIT:
                            finalize_shard()
                            shards_created += 1
                            if shards_created >= SHARDS_PER_RUN:
                                stop_run = True
                                success = True
                                break
                            shard_index += 1
                            env, shard_dir = open_env(shard_index)
                            txn = env.begin(write=True)
                            log(f"[SHARD] open new shard_{shard_index:04d}")

                        for mv in moves:
                            arr = board_to_tensor(board)
                            arr_u8 = np.asarray(arr, dtype=np.uint8)
                            tensor_bytes = arr_u8.tobytes()

                            move_idx = mv.from_square * 64 + mv.to_square
                            value = base_value if board.turn == chess.WHITE else -base_value

                            obj = {
                                "tensor": tensor_bytes,
                                "shape": list(arr_u8.shape),
                                "dtype": "uint8",
                                "label": {"policy": int(move_idx), "value": float(value)},
                            }

                            key = f"{global_index:012d}".encode()
                            txn.put(key, msgpack.packb(obj, use_bin_type=True))

                            samples_in_shard += 1
                            global_index += 1

                            if global_index % COMMIT_INTERVAL == 0:
                                txn.commit()
                                env.sync()
                                txn = env.begin(write=True)

                            board.push(mv)

                        success = True

                    except Exception as e:
                        retries += 1
                        log(f"[ERROR] game_idx={game_idx}, retry={retries}, err={e}")

                if not success:
                    log(f"[WARN] skip game_idx={game_idx}")

            pbar.close()

        finalize_shard()
        after = scan_and_clean_shards()
        total_after = sum(x[2] for x in after)

        save_checkpoint(
            last_game=game_idx if start_idx < total_games else total_games - 1,
            total_games=total_games,
            global_index=total_after,
            shard_index=after[-1][0] if after else shard_index,
        )

        log(f"[DONE] total samples={total_after}")
        log("===== RUN FINISHED =====")

    except Exception as e:
        log(f"[FATAL] 예외 발생: {e}")
        from parse_pgn_envcheck import env_diagnose
        env_diagnose(PGN_PATH, OUT_ROOT, TOOLS_DIR, PROJECT_ROOT, LOG_PATH)
        raise


if __name__ == "__main__":
    main()
