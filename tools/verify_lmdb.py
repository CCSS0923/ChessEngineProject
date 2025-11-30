import os
import json
import argparse
from typing import Dict, Any

import lmdb
import msgpack
import numpy as np


def load_meta(meta_path: str) -> Dict[str, Any]:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"[ERROR] meta.json 로드 실패: {meta_path}\n{e}")


def check_shard(shard_dir: str) -> None:
    print(f"\n=== Checking shard: {shard_dir} ===")

    # ---------- 파일 존재 검증 ----------
    data_path = os.path.join(shard_dir, "data.mdb")
    lock_path = os.path.join(shard_dir, "lock.mdb")
    meta_path = os.path.join(shard_dir, "meta.json")

    if not os.path.isfile(data_path):
        raise RuntimeError(f"[ERROR] data.mdb 없음: {data_path}")

    if not os.path.isfile(lock_path):
        print("[WARN] lock.mdb 없음 (문제는 아님)")

    if not os.path.isfile(meta_path):
        raise RuntimeError(f"[ERROR] meta.json 없음: {meta_path}")

    # ---------- meta.json 검증 ----------
    meta = load_meta(meta_path)

    print(f"  num_samples    : {meta.get('num_samples')}")
    print(f"  tensor_shape   : {meta.get('tensor_shape')}")
    print(f"  tensor_dtype   : {meta.get('tensor_dtype')}")
    print(f"  label_type     : {meta.get('label_type')}")

    if meta.get("tensor_shape") != [18, 8, 8]:
        raise RuntimeError("[ERROR] tensor_shape != [18,8,8]")

    if meta.get("tensor_dtype") != "uint8":
        raise RuntimeError("[ERROR] tensor_dtype != uint8")

    if meta.get("label_type") != "policy_value":
        raise RuntimeError("[ERROR] label_type != policy_value")

    num_samples = meta.get("num_samples", 0)
    if num_samples <= 0:
        raise RuntimeError("[ERROR] num_samples <= 0")

    print("  meta.json OK")

    # ---------- LMDB 내부 첫 key 검증 ----------
    env = lmdb.open(shard_dir, readonly=True, lock=False, readahead=False, max_readers=1)

    with env.begin() as txn:
        cur = txn.cursor()

        first_key = None
        first_val = None

        for k, v in cur:
            first_key = k
            first_val = v
            break

        if first_key is None:
            raise RuntimeError("[ERROR] shard 내부에 데이터가 없음")

    print(f"  First key: {first_key}")

    # ---------- msgpack 해제 ----------
    try:
        obj = msgpack.loads(first_val, raw=False)
    except Exception as e:
        raise RuntimeError(f"[ERROR] msgpack 해제 실패\n{e}")

    tensor_bytes = obj["tensor"]
    shape = obj["shape"]
    label = obj["label"]

    # ---------- 텐서 검증 ----------
    arr = np.frombuffer(tensor_bytes, dtype=np.uint8)
    arr = arr.reshape(shape)  # (18,8,8)

    print(f"  Tensor shape OK: {arr.shape}")
    print(f"  Tensor dtype  OK: {arr.dtype}")

    # 텐서 기본 통계
    print(f"  Tensor min/max/sum: {arr.min()} / {arr.max()} / {arr.sum()}")

    # ---------- 라벨 검증 ----------
    policy = label["policy"]
    value = label["value"]

    print(f"  policy = {policy}, value = {value}")

    if not (0 <= policy < 4096):
        raise RuntimeError("[ERROR] policy 범위 이상 (0~4095 아님)")

    if not (-1.0 <= value <= 1.0):
        raise RuntimeError("[ERROR] value 범위 이상 (-1~1 아님)")

    print("  Label OK")

    print("=== Shard OK ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_root", type=str, required=True)
    args = parser.parse_args()

    root = args.lmdb_root

    if not os.path.isdir(root):
        raise RuntimeError(f"[ERROR] LMDB root 디렉토리 없음: {root}")

    shard_names = sorted(x for x in os.listdir(root) if x.startswith("shard_"))

    if not shard_names:
        raise RuntimeError("[ERROR] shard_XXXX 폴더가 없음")

    print(f"[INFO] Found shards: {shard_names}")

    for name in shard_names:
        shard_dir = os.path.join(root, name)
        check_shard(shard_dir)

    print("\n=============================")
    print("ALL SHARDS VERIFIED SUCCESS!")
    print("=============================")


if __name__ == "__main__":
    main()
