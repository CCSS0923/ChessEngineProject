import lmdb
import msgpack
import os
import numpy as np

LMDB_PATH = r"D:\GoogleDrive\C++\ChessEngineProject\data\lmdb\standard_2025_01\shard_0000"

def main():
    env = lmdb.open(
        LMDB_PATH,
        subdir=os.path.isdir(LMDB_PATH),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    with env.begin(write=False) as txn:
        cur = txn.cursor()
        for k, v in cur:
            print("KEY:", k)
            obj = msgpack.unpackb(v, raw=False)
            print("DICT KEYS:", obj.keys())

            tensor_raw = obj["tensor"]
            shape = obj["shape"]
            dtype = obj["dtype"]
            label = obj["label"]

            print("shape:", shape)
            print("dtype:", dtype)
            print("label type:", type(label))
            print("label content:", label)

            # 텐서 내용 간단 요약
            arr = np.frombuffer(tensor_raw, dtype=np.uint8).reshape(shape)
            print("tensor ndim:", arr.ndim)
            print("tensor shape:", arr.shape)
            print("tensor min/max:", arr.min(), arr.max())

            # 각 채널별 1의 개수 (처음 몇 개만)
            counts = arr.reshape(arr.shape[0], -1).sum(axis=1)
            for i, c in enumerate(counts):
                print(f"channel {i}: sum={int(c)}")
            break  # 첫 샘플만

if __name__ == "__main__":
    main()
