import os
import sys
import json
import math
import argparse
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import lmdb
import msgpack
import numpy as np
from tqdm import tqdm


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def find_shard_dirs(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise SystemExit(f"LMDB root not found: {root}")
    names = sorted(
        d for d in os.listdir(root)
        if d.startswith("shard_") and os.path.isdir(os.path.join(root, d))
    )
    return [os.path.join(root, d) for d in names]


def load_meta(shard_dir: str) -> Optional[Dict[str, Any]]:
    meta_path = os.path.join(shard_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[WARN] meta.json decode failed in {shard_dir}: {e}")
        return None


def open_env(shard_dir: str) -> lmdb.Environment:
    env = lmdb.open(
        shard_dir,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=128,
        subdir=True,
    )
    return env


def is_close_value(x: float, targets=( -1.0, 0.0, 1.0 ), eps: float = 1e-6) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    for t in targets:
        if abs(v - t) <= eps:
            return True
    return False


def check_sample_structure(
    key: bytes,
    value: bytes,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        obj = msgpack.unpackb(value, raw=False)
    except Exception as e:
        errors.append(f"msgpack decode error: {e}")
        return False, errors

    if not isinstance(obj, dict):
        errors.append(f"top-level object is not dict: {type(obj)}")
        return False, errors

    # ---- board tensor 찾기 (planes / x / board / input 등 허용) ----
    board = None
    board_key = None
    for cand in ("planes", "x", "board", "input"):
        if cand in obj:
            board = obj[cand]
            board_key = cand
            break

    if board is None:
        errors.append("missing board tensor (planes/x/board/input)")
    else:
        arr = np.asarray(board)
        if arr.shape != (18, 8, 8):
            errors.append(f"board tensor shape invalid: {arr.shape}, expected (18, 8, 8)")
        if arr.dtype != np.uint8:
            # dtype은 엄격하게 uint8 요구
            errors.append(f"board tensor dtype invalid: {arr.dtype}, expected uint8")

    # ---- label: policy / value ----
    if "policy" not in obj:
        errors.append("missing 'policy'")
    else:
        policy = obj["policy"]
        if not isinstance(policy, int):
            errors.append(f"'policy' not int: {type(policy)}")
        else:
            if not (0 <= policy < 4096):
                errors.append(f"'policy' out of range: {policy}")

    if "value" not in obj:
        errors.append("missing 'value'")
    else:
        value_field = obj["value"]
        if not isinstance(value_field, (int, float)):
            errors.append(f"'value' not number: {type(value_field)}")
        else:
            if not is_close_value(value_field):
                errors.append(f"'value' not in {{-1,0,1}}: {value_field}")

    ok = len(errors) == 0
    return ok, errors


def analyze_keys(keys: List[bytes]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "total_keys": len(keys),
        "all_digit": False,
        "min_id": None,
        "max_id": None,
        "missing_count": 0,
        "missing_examples": [],
        "duplicate_count": 0,
        "duplicate_examples": [],
    }

    if not keys:
        return info

    # 모든 key가 digit string인지 확인
    try:
        decoded = [k.decode("ascii") for k in keys]
    except UnicodeDecodeError:
        return info  # 연속성 검사 불가

    if not all(s.isdigit() for s in decoded):
        return info

    info["all_digit"] = True

    nums = sorted(int(s) for s in decoded)
    info["min_id"] = nums[0]
    info["max_id"] = nums[-1]

    # 연속성 검사
    expected_len = info["max_id"] - info["min_id"] + 1
    # 중복/누락 체크
    seen = set()
    duplicates = []
    for n in nums:
        if n in seen:
            duplicates.append(n)
        else:
            seen.add(n)

    missing = []
    if expected_len != len(nums) or duplicates:
        # 누락된 id 찾기 (큰 샤드에서 비용이 크지만, 한 번만 수행)
        for nid in range(info["min_id"], info["max_id"] + 1):
            if nid not in seen:
                missing.append(nid)

    info["duplicate_count"] = len(duplicates)
    info["duplicate_examples"] = duplicates[:10]
    info["missing_count"] = len(missing)
    info["missing_examples"] = missing[:10]

    return info


def check_shard(shard_dir: str, isolate_root: Optional[str]) -> Dict[str, Any]:
    shard_name = os.path.basename(shard_dir)
    log(f"Checking shard: {shard_name}")

    meta = load_meta(shard_dir)
    meta_num_samples = None
    if meta is not None:
        meta_num_samples = meta.get("num_samples", None)

    env = open_env(shard_dir)
    stat = env.stat()
    n_entries = stat.get("entries", 0)

    result: Dict[str, Any] = {
        "shard": shard_name,
        "path": shard_dir,
        "entries": n_entries,
        "meta_num_samples": meta_num_samples,
        "meta_match": None,
        "structure_ok": True,
        "structure_errors": 0,
        "structure_error_examples": [],
        "key_info": {},
        "has_error": False,
    }

    keys: List[bytes] = []
    structure_error_examples: List[Dict[str, Any]] = []

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        total = n_entries
        if total == 0:
            log(f"[WARN] shard {shard_name} has 0 entries")
        iter_cursor = cursor.iternext(keys=True, values=True)
        if total is not None and total > 0:
            iter_cursor = tqdm(iter_cursor, total=total, desc=f"{shard_name}", unit="sample")

        for k, v in iter_cursor:
            keys.append(k)
            ok, errs = check_sample_structure(k, v)
            if not ok:
                result["structure_ok"] = False
                result["structure_errors"] += 1
                if len(structure_error_examples) < 20:
                    try:
                        k_str = k.decode("ascii", errors="replace")
                    except Exception:
                        k_str = repr(k)
                    structure_error_examples.append({
                        "key": k_str,
                        "errors": errs,
                    })

    result["structure_error_examples"] = structure_error_examples

    # meta.json과 entry 수 비교
    if meta_num_samples is not None:
        result["meta_match"] = (meta_num_samples == n_entries)

    # key 연속성/숫자 여부 분석
    key_info = analyze_keys(keys)
    result["key_info"] = key_info

    # 에러 판단
    has_key_problem = (
        key_info.get("all_digit", False) and
        (key_info.get("missing_count", 0) > 0 or key_info.get("duplicate_count", 0) > 0)
    )
    has_meta_problem = (meta_num_samples is not None and not result["meta_match"])
    has_struct_problem = not result["structure_ok"]

    result["has_error"] = bool(has_key_problem or has_meta_problem or has_struct_problem)

    # 필요 시 shard 격리
    if result["has_error"] and isolate_root is not None:
        os.makedirs(isolate_root, exist_ok=True)
        target = os.path.join(isolate_root, shard_name)
        if not os.path.exists(target):
            log(f"[ACTION] moving corrupt shard {shard_name} -> {target}")
            shutil.move(shard_dir, target)
        else:
            log(f"[WARN] corrupt target already exists, skip move: {target}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LMDB Integrity Checker for ChessEngineProject shards"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="LMDB dataset root (e.g. data/lmdb/standard_2025_01)",
    )
    parser.add_argument(
        "--isolate-corrupt",
        action="store_true",
        help="Move shards with errors into <root>/corrupt/",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="integrity_report.json",
        help="Output report JSON filename (saved under --root)",
    )

    args = parser.parse_args()
    root = os.path.abspath(args.root)
    isolate_root = os.path.join(root, "corrupt") if args.isolate_corrupt else None

    log(f"LMDB root: {root}")
    if isolate_root is not None:
        log(f"Corrupt shard isolation dir: {isolate_root}")

    shard_dirs = find_shard_dirs(root)
    if not shard_dirs:
        raise SystemExit(f"No shard_* directories found under: {root}")

    log(f"Found {len(shard_dirs)} shard(s).")

    all_results: List[Dict[str, Any]] = []
    total_entries = 0
    total_errors = 0
    shards_with_error = 0

    for shard_dir in shard_dirs:
        res = check_shard(shard_dir, isolate_root)
        all_results.append(res)
        total_entries += res.get("entries", 0)
        total_errors += res.get("structure_errors", 0)
        if res.get("has_error", False):
            shards_with_error += 1

        # shard별 요약 출력
        key_info = res["key_info"]
        log(
            f"[SUMMARY] {res['shard']}: "
            f"entries={res['entries']}, "
            f"meta_num_samples={res['meta_num_samples']}, "
            f"meta_match={res['meta_match']}, "
            f"structure_ok={res['structure_ok']}, "
            f"key_all_digit={key_info.get('all_digit', False)}, "
            f"missing_keys={key_info.get('missing_count', 0)}, "
            f"duplicate_keys={key_info.get('duplicate_count', 0)}, "
            f"has_error={res['has_error']}"
        )

    # 전체 요약
    all_ok = (shards_with_error == 0)
    log("============================================================")
    log(f"Total shards      : {len(shard_dirs)}")
    log(f"Total entries     : {total_entries}")
    log(f"Shards with error : {shards_with_error}")
    log(f"Total struct errs : {total_errors}")
    log(f"DATASET OK        : {all_ok}")
    log("============================================================")

    # JSON 리포트 저장
    report_path = os.path.join(root, args.report)
    report_obj = {
        "root": root,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "all_ok": all_ok,
        "total_shards": len(shard_dirs),
        "total_entries": total_entries,
        "shards_with_error": shards_with_error,
        "total_structure_errors": total_errors,
        "shards": all_results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, indent=2, ensure_ascii=False)
    log(f"Report written to: {report_path}")

    # 종료 코드: OK면 0, 에러 있으면 1
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
