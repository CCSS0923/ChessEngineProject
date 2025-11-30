#include "search.h"
#include "eval.h"
#include "movegen.h"
#include "types.h"
#include <limits>
#include <vector>

namespace chess {

static const int INF = 1000000000;
static const int MATE_SCORE = 1000000;

//---------------------------------------------
// 현재 보드에서 side의 킹이 체크인지 여부 확인
//---------------------------------------------
static bool inCheck(const Board &b, Color side) {
  int kingSq = -1;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = b.pieceAt(sq);
    if (p != NO_PIECE && pieceType(p) == KING &&
        isWhite(p) == (side == WHITE)) {
      kingSq = sq;
      break;
    }
  }
  if (kingSq == -1)
    return false;

  return b.isSquareAttacked(kingSq, opposite(side));
}

//---------------------------------------------
// negamax: side-to-move 기준 점수 계산
//---------------------------------------------
static int negamax(Board &b, int depth, int alpha, int beta) {
  Color us = b.stm;

  // -------------------------------
  // 리프 노드
  // -------------------------------
  if (depth == 0) {
    // NN 기반 평가 함수: 화이트 기준 score 반환
    int score = evaluate(b);

    // side-to-move 기준 부호 변환
    if (us == BLACK)
      score = -score;

    return score;
  }

  // -------------------------------
  // 합법수 생성
  // -------------------------------
  std::vector<Move> moves;
  MoveGen::generate(b, moves);

  // -------------------------------
  // 이동 불가 → 체크메이트 or 스테일메이트
  // -------------------------------
  if (moves.empty()) {
    if (inCheck(b, us)) {
      // 현재 side-to-move가 체크메이트 당한 상태
      return -MATE_SCORE;
    } else {
      // 스테일메이트
      return 0;
    }
  }

  // -------------------------------
  // Alpha-Beta Negamax 루프
  // -------------------------------
  int best = -INF;

  for (const Move &m : moves) {
    Board child = b;
    child.makeMove(m);

    int score = -negamax(child, depth - 1, -beta, -alpha);

    if (score > best)
      best = score;

    if (score > alpha)
      alpha = score;

    if (alpha >= beta)
      break; // 컷
  }

  return best;
}

//---------------------------------------------
// searchBest: depth 수만큼 탐색 후 최선수 반환
//---------------------------------------------
Move searchBest(Board &b, int depth) {
  std::vector<Move> moves;
  MoveGen::generate(b, moves);

  if (moves.empty())
    return Move(); // 불가능하면 null move 반환

  Color us = b.stm;

  int alpha = -INF;
  int beta = INF;
  int bestScore = -INF;
  Move bestMove = moves[0];

  for (const Move &m : moves) {
    Board child = b;
    child.makeMove(m);

    int score = -negamax(child, depth - 1, -beta, -alpha);

    if (score > bestScore) {
      bestScore = score;
      bestMove = m;
    }

    if (score > alpha)
      alpha = score;
  }

  return bestMove;
}

} // namespace chess
