import os
import sys
import json
import mmap
import lmdb
import time
import subprocess
from datetime import datetime

# ----------------------------
# 경로 설정
# ----------------------------
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TOOLS_DIR, ".."))

PGN_PATH = os.path.join(PROJECT_ROOT, "data", "pgn_raw", "lichess_db_standard_rated_2025-01.pgn")
OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "lmdb", "standard_2025_01")

# ----------------------------
# 출력 함수
# ----------------------------
def info(msg):
    print(f"[INFO] {msg}")

def warn(msg):
    print(f"[WARN] {msg}")

def err(msg):
    print(f"[ERROR] {msg}")

# ----------------------------
# 1) 정상 경로인지 점검
# ----------------------------
def check_paths():
    info("=== 1) 경로 점검 ===")
    print(f"PGN_PATH     = {PGN_PATH}")
    print(f"OUT_ROOT     = {OUT_ROOT}")
    print(f"WORKING_DIR  = {os.getcwd()}")
    print(f"TOOLS_DIR    = {TOOLS_DIR}")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")

    if not os.path.exists(PGN_PATH):
        err("PGN 파일 없음")
    else:
        info("PGN 파일 존재 OK")

    if not os.path.exists(OUT_ROOT):
        warn("OUT_ROOT 디렉터리가 없음 → 자동 생성 필요")
    else:
        info("OUT_ROOT 디렉터리 존재 OK")

# ----------------------------
# 2) OUT_ROOT 권한 점검
# ----------------------------
def check_permissions():
    info("=== 2) OUT_ROOT 권한 점검 ===")
    try:
        test_file = os.path.join(OUT_ROOT, "permission_test.txt")
        os.makedirs(OUT_ROOT, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
        info("쓰기 권한 OK")
    except Exception as e:
        err(f"쓰기 불가: {e}")

# ----------------------------
# 3) shard 디렉터리 점검
# ----------------------------
def check_shards():
    info("=== 3) shard 디렉터리 점검 ===")

    count = 0
    if not os.path.exists(OUT_ROOT):
        warn("OUT_ROOT 없음 → shard 없음")
        return

    for d in os.listdir(OUT_ROOT):
        if d.startswith("shard_") and len(d) == 10:
            shard_dir = os.path.join(OUT_ROOT, d)
            meta_path = os.path.join(shard_dir, "meta.json")
            count += 1

            print(f" - {d}")
            if os.path.exists(meta_path):
                print("   meta.json OK")
            else:
                print("   meta.json 없음 → 미완성 shard")
                print("   data.mdb 존재:", os.path.exists(os.path.join(shard_dir, "data.mdb")))

    if count == 0:
        info("shard 없음 (정상)")

# ----------------------------
# 4) data.mdb 크기 점검
# ----------------------------
def check_large_files():
    info("=== 4) data.mdb 크기 점검 ===")

    if not os.path.exists(OUT_ROOT):
        return

    for d in os.listdir(OUT_ROOT):
        if d.startswith("shard_") and len(d) == 10:
            shard_dir = os.path.join(OUT_ROOT, d)
            data_file = os.path.join(shard_dir, "data.mdb")
            if os.path.exists(data_file):
                size = os.path.getsize(data_file)
                print(f"{d} → data.mdb = {size/1024/1024/1024:.2f} GB")
                if size >= 64 * 1024 * 1024 * 1024:
                    warn("  64GB 이상 → 첫 게임에서 즉시 need_roll=True 발생 원인")

# ----------------------------
# 5) LMDB 환경이 초기화 가능한지 점검
# ----------------------------
def test_lmdb_open():
    info("=== 5) LMDB 환경 시험(LMDB 열기) ===")

    test_dir = os.path.join(OUT_ROOT, "test_env")
    try:
        os.makedirs(test_dir, exist_ok=True)
        env = lmdb.open(
            test_dir,
            map_size=1 << 30,
            subdir=True,
            readonly=False,
            lock=True,
            meminit=False,
            readahead=False,
            map_async=False,
        )
        txn = env.begin(write=True)
        txn.put(b"test", b"123")
        txn.commit()
        env.sync()
        env.close()
        info("LMDB open/write/close OK")
    except Exception as e:
        err(f"LMDB open 실패: {e}")

# ----------------------------
# 6) parse_pgn.py 실제 실행 경로 점검
# ----------------------------
def verify_current_script():
    info("=== 6) parse_pgn.py 실행 위치 점검 ===")

    try:
        import parse_pgn
        print("parse_pgn.py 실제 로딩 경로:", parse_pgn.__file__)
    except Exception as e:
        err(f"로드 실패: {e}")

# ----------------------------
# 7) PGN 파일 정상 읽기 점검
# ----------------------------
def check_pgn_read():
    info("=== 7) PGN 파일 읽기 점검 ===")

    try:
        with open(PGN_PATH, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            pos = mm.find(b"[Event ")
            if pos == -1:
                warn("첫 Event 태그를 찾을 수 없음")
            else:
                info(f"[Event ] offset={pos}")
            mm.close()
    except Exception as e:
        err(f"PGN mmap 실패: {e}")

# ----------------------------
# 8) Windows 락 확인 - handle 검사
# ----------------------------
def check_windows_handles():
    info("=== 8) Windows 파일 잠금(handle) 검사 ===")
    # Handle 검사: 열린 프로세스를 보여줌
    try:
        subprocess.run(["handle.exe", OUT_ROOT], check=False)
        info("handle.exe 결과 확인 필요")
    except Exception:
        warn("handle.exe 사용 불가 (SysInternals 설치 필요)")

# ----------------------------
# 실행
# ----------------------------
if __name__ == "__main__":
    check_paths()
    check_permissions()
    check_shards()
    check_large_files()
    test_lmdb_open()
    verify_current_script()
    check_pgn_read()
    check_windows_handles()
