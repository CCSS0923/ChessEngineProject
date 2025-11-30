import lmdb
import chess.pgn
import msgpack
from tqdm import tqdm
from fen_to_tensor import board_to_tensor
from move_index import uci_to_index
from config import PGN_PATH, LMDB_PATH

MAP_SIZE = 200 * 1024 * 1024 * 1024

def main():
    env = lmdb.open(
        LMDB_PATH,
        map_size=MAP_SIZE,
        subdir=True,
        lock=False,
        readahead=False
    )

    idx = 0
    with open(PGN_PATH, encoding="utf-8") as pgn, env.begin(write=True) as txn:
        pbar = tqdm(unit="moves")
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                tensor = board_to_tensor(board)
                tensor_bytes = tensor.tobytes()
                label = uci_to_index(move)
                sample = {
                    "tensor": tensor_bytes,
                    "shape": tensor.shape,
                    "dtype": str(tensor.dtype),
                    "label": label
                }
                key = f"{idx:012d}".encode()
                val = msgpack.dumps(sample, use_bin_type=True)
                txn.put(key, val)
                idx += 1
                board.push(move)
                pbar.update(1)
                if idx % 50000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
        pbar.close()
    print("total samples:", idx)

if __name__ == "__main__":
    main()
