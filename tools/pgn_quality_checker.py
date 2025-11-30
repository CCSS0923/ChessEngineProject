# 실행 예 :
# python pgn_quality_checker.py --pgn "D:\data\pgn_raw\lichess_db_standard_rated_2025-01.pgn" --sample_rate 1.0
# (샘플링으로 속도 조절 가능: 예 --sample_rate 0.1 → 10%만 분석)

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import chess
import chess.pgn
from tqdm import tqdm


@dataclass
class PGNStats:
    total_games: int = 0          # 전체 게임 수 (샘플링 포함)
    sampled_games: int = 0        # 샘플링 후 실제로 분석한 게임 수

    parse_errors: int = 0         # python-chess로 파싱 실패
    illegal_games: int = 0        # mainline 생성 중 예외(불법 수 등)

    no_result_tag: int = 0        # Result 태그 없음
    result_1_0: int = 0
    result_0_1: int = 0
    result_half: int = 0          # 1/2-1/2
    result_other: int = 0         # '*', 기타

    total_plies: int = 0          # 전체 half-move 합
    min_plies: int = 1_000_000    # 최솟값
    max_plies: int = 0            # 최댓값

    short_games_plies_lt_8: int = 0   # 8 half-moves 미만 (4수 미만)
    short_games_plies_lt_16: int = 0  # 16 half-moves 미만 (8수 미만)

    # 특정 ECO 코드/Variant 등 나중에 확장 가능


def classify_result(result: str, stats: PGNStats) -> None:
    if not result:
        stats.no_result_tag += 1
        stats.result_other += 1
        return

    if result == "1-0":
        stats.result_1_0 += 1
    elif result == "0-1":
        stats.result_0_1 += 1
    elif result == "1/2-1/2":
        stats.result_half += 1
    else:
        stats.result_other += 1


def update_move_stats(num_plies: int, stats: PGNStats) -> None:
    stats.total_plies += num_plies
    if num_plies < stats.min_plies:
        stats.min_plies = num_plies
    if num_plies > stats.max_plies:
        stats.max_plies = num_plies

    if num_plies < 8:
        stats.short_games_plies_lt_8 += 1
        stats.short_games_plies_lt_16 += 1
    elif num_plies < 16:
        stats.short_games_plies_lt_16 += 1


def analyze_game(game: chess.pgn.Game, stats: PGNStats) -> None:
    # 결과 태그
    result = game.headers.get("Result", "")
    classify_result(result, stats)

    # 합법 수 / half-move 수 계산
    try:
        board = game.board()
        plies = 0
        for move in game.mainline_moves():
            board.push(move)
            plies += 1
    except Exception:
        stats.illegal_games += 1
        return

    update_move_stats(plies, stats)


def pretty_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0 (0.00%)"
    return f"{count} ({count / total * 100:.2f}%)"


def print_report(stats: PGNStats) -> None:
    print("\n===== PGN QUALITY REPORT =====")
    print(f"Total games in file (scanned)     : {stats.total_games}")
    print(f"Sampled games (actually analyzed) : {stats.sampled_games}")
    print()
    print(f"Parse errors                      : {pretty_ratio(stats.parse_errors, stats.total_games)}")
    print(f"Illegal/invalid games             : {pretty_ratio(stats.illegal_games, stats.sampled_games)}")
    print()
    print("Result distribution (sampled games):")
    print(f"  1-0         : {pretty_ratio(stats.result_1_0, stats.sampled_games)}")
    print(f"  0-1         : {pretty_ratio(stats.result_0_1, stats.sampled_games)}")
    print(f"  1/2-1/2     : {pretty_ratio(stats.result_half, stats.sampled_games)}")
    print(f"  other(*)    : {pretty_ratio(stats.result_other, stats.sampled_games)}")
    print(f"  no Result tag: {pretty_ratio(stats.no_result_tag, stats.sampled_games)}")
    print()
    if stats.sampled_games > 0:
        avg_plies = stats.total_plies / stats.sampled_games
    else:
        avg_plies = 0.0
    print("Game length (in half-moves / plies):")
    print(f"  min plies   : {0 if stats.max_plies == 0 else stats.min_plies}")
    print(f"  max plies   : {stats.max_plies}")
    print(f"  avg plies   : {avg_plies:.2f}")
    print()
    print("Short games:")
    print(f"  plies < 8   (4 moves) : {pretty_ratio(stats.short_games_plies_lt_8, stats.sampled_games)}")
    print(f"  plies < 16 (8 moves)  : {pretty_ratio(stats.short_games_plies_lt_16, stats.sampled_games)}")
    print("================================\n")


def analyze_pgn(
    pgn_path: str,
    max_games: Optional[int] = None,
    sample_rate: float = 1.0,
) -> PGNStats:
    """
    sample_rate: 1.0 → 전체 게임 분석
                 0.1 → 10%만 랜덤 샘플링 (간단 랜덤 스킵)
    max_games: 최대 분석 게임 수 (None 이면 제한 없음)
    """
    import random

    stats = PGNStats()

    filesize = os.path.getsize(pgn_path)
    print(f"[INFO] PGN file: {pgn_path}")
    print(f"[INFO] File size: {filesize / (1024**3):.2f} GB")
    print(f"[INFO] sample_rate={sample_rate}, max_games={max_games}")
    print("[INFO] Scanning PGN... (streaming)")

    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        # tqdm는 전체 게임 수를 모르므로 dynamic 모드로 사용
        pbar = tqdm(total=None, unit="game")
        while True:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                stats.parse_errors += 1
                continue

            if game is None:
                break

            stats.total_games += 1
            pbar.update(1)

            # max_games 제한
            if max_games is not None and stats.total_games > max_games:
                break

            # 샘플링
            if sample_rate < 1.0:
                r = random.random()
                if r > sample_rate:
                    continue  # 이 게임은 건너뜀

            stats.sampled_games += 1
            analyze_game(game, stats)

        pbar.close()

    return stats


def parse_args():
    parser = argparse.ArgumentParser(description="PGN Quality Analyzer")
    parser.add_argument(
        "--pgn",
        type=str,
        required=True,
        help="PGN 파일 경로 (예: D:\\data\\pgn_raw\\lichess_db_standard_rated_2025-01.pgn)",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=None,
        help="최대 분석 게임 수 (기본: 전체)",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="샘플링 비율 (0.0~1.0, 예: 0.1 → 10%만 분석)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.pgn):
        print(f"[ERROR] PGN 파일을 찾을 수 없음: {args.pgn}")
        sys.exit(1)

    if not (0.0 < args.sample_rate <= 1.0):
        print("[ERROR] sample_rate는 0.0 < r <= 1.0 이어야 함")
        sys.exit(1)

    stats = analyze_pgn(
        pgn_path=args.pgn,
        max_games=args.max_games,
        sample_rate=args.sample_rate,
    )

    print_report(stats)


if __name__ == "__main__":
    main()
