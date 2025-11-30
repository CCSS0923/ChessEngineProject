# 실행 예시 :
# 전체 샤드 검사 (완전 검사):
# python tools/lmdb_integrity_checker.py --lmdb_root data/lmdb/standard_2025_01
# 큰 데이터에서는 너무 오래 걸릴 수 있으므로 샘플 검사:
# python tools/lmdb_integrity_checker.py --lmdb_root data/lmdb/standard_2025_01 --max_samples 2000
# 특정 개수 샤드만 확인:
# python tools/lmdb_integrity_checker.py --lmdb_root data/lmdb/standard_2025_01 --max_shards 5

import argparse
import os
import sys
import json
import lmdb
import msgpack
from tqdm import tqdm
import numpy as np


EXPECTED_SHAPE = [18, 8, 8]
EXPECTED_DTYPE = "uint8"
EXPECTED_TENSOR_BYTES = 18 * 8 * 8  # 1152 bytes
EXPECTED_POLICY_SIZE = 4096


class LMDBShardReport:
    def __init__(self, shard_name):
        self.shard_name = shard_name
        self.ok = True
        self.errors = []
        self.num_samples = 0
        self.checked_samples = 0
        self.missing_keys = 0
        self.invalid_tensor = 0
        self.invalid_label = 0

    def add_error(self, msg):
        self.ok = False
        self.errors.append(msg)


def check_shard(shard_path, shard_name, max_samples=None):
    report = LMDBShardReport(shard_name)
    meta_path = os.path.join(shard_path, "meta.json")
    data_file = os.path.join(shard_path, "data.mdb")

    # meta.json 체크
    if not os.path.exists(meta_path):
        report.add_error("meta.json missing (incomplete shard)")
        return report

    # meta.json 읽기
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        num_samples = int(meta.get("num_samples", -1))
    except Exception as e:
        report.add_error(f"meta.json invalid: {e}")
        return report

    if num_samples < 0:
        report.add_error("invalid num_samples in meta.json")
        return report

    report.num_samples = num_samples

    # data.mdb 체크
    if not os.path.exists(data_file):
        report.add_error("data.mdb missing")
        return report

    # LMDB 열기
    try:
        env = lmdb.open(
            shard_path,
            readonly=True,
            lock=False,
            max_readers=16,
            subdir=True,
            readahead=False,
        )
    except Exception as e:
        report.add_error(f"lmdb open failed: {e}")
        return report

    # key range 검사
    with env.begin(write=False) as txn:
        for i in tqdm(range(num_samples), desc=f"Checking {shard_name}", unit="sample"):
            if max_samples and i >= max_samples:
                break

            key = f"{i:012d}".encode("ascii")
            buf = txn.get(key)
            if buf is None:
                report.missing_keys += 1
                report.ok = False
                continue

            report.checked_samples += 1

            try:
                obj = msgpack.unpackb(buf, raw=False)
            except Exception:
                report.invalid_tensor += 1
                report.ok = False
                continue

            # tensor 검사
            t_bytes = obj.get("tensor", None)
            shape = obj.get("shape", None)
            dtype = obj.get("dtype", None)
            lbl = obj.get("label", None)

            if (
                t_bytes is None
                or shape != EXPECTED_SHAPE
                or dtype != EXPECTED_DTYPE
                or len(t_bytes) != EXPECTED_TENSOR_BYTES
            ):
                report.invalid_tensor += 1
                report.ok = False

            # label 검사
            if not isinstance(lbl, dict):
                report.invalid_label += 1
                report.ok = False
            else:
                p = lbl.get("policy", None)
                v = lbl.get("value", None)
                if not isinstance(p, int) or not (0 <= p < EXPECTED_POLICY_SIZE):
                    report.invalid_label += 1
                    report.ok = False
                if not isinstance(v, (float, int)) or not (-1.5 <= v <= 1.5):
                    report.invalid_label += 1
                    report.ok = False

    return report


def run_check(lmdb_root, max_samples=None, max_shards=None):
    if not os.path.isdir(lmdb_root):
        print(f"[ERROR] LMDB root not found: {lmdb_root}")
        return

    shard_names = sorted(
        d for d in os.listdir(lmdb_root)
        if d.startswith("shard_") and os.path.isdir(os.path.join(lmdb_root, d))
    )

    if max_shards:
        shard_names = shard_names[:max_shards]

    reports = []
    for name in shard_names:
        path = os.path.join(lmdb_root, name)
        rep = check_shard(path, name, max_samples=max_samples)
        reports.append(rep)

    print("\n===== LMDB INTEGRITY REPORT =====")
    for rep in reports:
        status = "OK" if rep.ok else "BAD"
        print(f"Shard {rep.shard_name} : {status}")
        print(f"  num_samples      : {rep.num_samples}")
        print(f"  checked_samples  : {rep.checked_samples}")
        print(f"  missing_keys     : {rep.missing_keys}")
        print(f"  invalid_tensor   : {rep.invalid_tensor}")
        print(f"  invalid_label    : {rep.invalid_label}")
        if rep.errors:
            for e in rep.errors:
                print(f"  ERROR: {e}")
        print("------------------------------")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb_root", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=None, help="per-shard sample limit")
    p.add_argument("--max_shards", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_check(args.lmdb_root, args.max_samples, args.max_shards)


if __name__ == "__main__":
    main()
