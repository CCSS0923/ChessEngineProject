import os
import json
import lmdb
import mmap
import traceback
from datetime import datetime


def env_diagnose(pgn_path: str, out_root: str, tools_dir: str, project_root: str, log_path: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    def log(x):
        line = f"[{ts}] [ENVCHK] {x}"
        print(line)
        lines.append(line)

    log("=== 자동 진단 시작 ===")

    # 1) 경로 점검
    log(f"PGN_PATH={pgn_path}")
    log(f"OUT_ROOT={out_root}")
    log(f"TOOLS_DIR={tools_dir}")
    log(f"PROJECT_ROOT={project_root}")

    # 2) PGN 존재 여부
    if not os.path.exists(pgn_path):
        log("[ERR] PGN 파일 없음")
    else:
        log("[OK] PGN 파일 존재")

    # 3) OUT_ROOT 확인
    if not os.path.exists(out_root):
        log("[ERR] OUT_ROOT 없음")
    else:
        log("[OK] OUT_ROOT 존재")
        # 권한 체크
        try:
            testfile = os.path.join(out_root, ".__test_write")
            with open(testfile, "w") as f:
                f.write("x")
            os.remove(testfile)
            log("[OK] OUT_ROOT 쓰기 가능")
        except Exception as e:
            log(f"[ERR] OUT_ROOT 쓰기 실패: {e}")

    # 4) LMDB 환경 오픈 테스트
    try:
        test_env = lmdb.open(out_root, map_size=1 << 30, subdir=False, readonly=False)
        with test_env.begin(write=True) as txn:
            txn.put(b"envchk", b"1")
        test_env.close()
        log("[OK] LMDB open/write/close 정상")
    except Exception as e:
        log(f"[ERR] LMDB 테스트 실패: {e}")

    # 5) shard 디렉터리 검사
    if os.path.exists(out_root):
        shards = [d for d in os.listdir(out_root) if d.startswith("shard_")]
        if shards:
            log(f"[WARN] 기존 shard 존재: {shards}")
            for sd in shards:
                sp = os.path.join(out_root, sd, "data.mdb")
                if os.path.exists(sp):
                    log(f"[OK] {sd} data.mdb size={os.path.getsize(sp)}")
                else:
                    log(f"[ERR] {sd} data.mdb 없음")
        else:
            log("[OK] shard 없음")

    # 6) PGN mmap 테스트
    if os.path.exists(pgn_path):
        try:
            with open(pgn_path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                pos = mm.find(b"[Event ")
                if pos != -1:
                    log(f"[OK] PGN mmap 정상, 첫 [Event] offset={pos}")
                else:
                    log("[ERR] PGN 파일이 손상되었을 가능성")
                mm.close()
        except Exception as e:
            log(f"[ERR] PGN mmap 실패: {e}")

    # 7) parse_pgn.py import 경로 점검
    try:
        import parse_pgn
        log(f"[OK] parse_pgn.py import OK → {parse_pgn.__file__}")
    except Exception as e:
        log(f"[ERR] parse_pgn import 실패: {e}")

    # 8) traceback 출력 (원래 오류)
    last_trace = traceback.format_exc()
    log("[INFO] 직전 예외 stacktrace:")
    for line in last_trace.split("\n"):
        log("  " + line)

    # 로그 저장
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for l in lines:
                f.write(l + "\n")
    except:
        pass

    log("=== 자동 진단 종료 ===")
