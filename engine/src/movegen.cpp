#include "movegen.h"
#include "board.h"
#include "move.h"
#include <cstdlib>
#include <vector>

namespace chess {

static bool isOnBoard(int sq) { return sq >= 0 && sq < 64; }

static void addMove(std::vector<Move> &out, int from, int to,
                    Piece promo = NO_PIECE) {
  out.emplace_back(from, to, promo);
}

// 폰 pseudo-legal
static void genPawnMoves(const Board &b, int sq, std::vector<Move> &out) {
  Piece p = b.pieceAt(sq);
  if (p == NO_PIECE)
    return;
  Color us = b.stm;
  if (isWhite(p) != (us == WHITE))
    return;

  int file = sq % 8;
  int rank = sq / 8;
  int dir = (us == WHITE) ? 1 : -1;

  int nextRank = rank + dir;
  if (nextRank >= 0 && nextRank < 8) {
    int oneStep = nextRank * 8 + file;

    // 앞으로 한 칸 (빈 칸)
    if (b.pieceAt(oneStep) == NO_PIECE) {
      // 프로모션 여부
      if (nextRank == 0 || nextRank == 7) {
        // 4종류 프로모션
        if (us == WHITE) {
          addMove(out, sq, oneStep, WQ);
          addMove(out, sq, oneStep, WR);
          addMove(out, sq, oneStep, WB);
          addMove(out, sq, oneStep, WN);
        } else {
          addMove(out, sq, oneStep, BQ);
          addMove(out, sq, oneStep, BR);
          addMove(out, sq, oneStep, BB);
          addMove(out, sq, oneStep, BN);
        }
      } else {
        addMove(out, sq, oneStep);
      }

      // 처음 위치에서 두 칸
      int startRank = (us == WHITE) ? 1 : 6;
      if (rank == startRank) {
        int twoRank = rank + 2 * dir;
        int twoStep = twoRank * 8 + file;
        if (twoRank >= 0 && twoRank < 8 && b.pieceAt(twoStep) == NO_PIECE) {
          addMove(out, sq, twoStep);
        }
      }
    }

    // 대각 캡처
    for (int df = -1; df <= 1; df += 2) {
      int cf = file + df;
      if (cf < 0 || cf >= 8)
        continue;
      int cr = nextRank;
      int to = cr * 8 + cf;
      Piece target = b.pieceAt(to);
      if (target != NO_PIECE && isWhite(target) != (us == WHITE)) {
        // 프로모션?
        if (cr == 0 || cr == 7) {
          if (us == WHITE) {
            addMove(out, sq, to, WQ);
            addMove(out, sq, to, WR);
            addMove(out, sq, to, WB);
            addMove(out, sq, to, WN);
          } else {
            addMove(out, sq, to, BQ);
            addMove(out, sq, to, BR);
            addMove(out, sq, to, BB);
            addMove(out, sq, to, BN);
          }
        } else {
          addMove(out, sq, to);
        }
      }
    }

    // 앙파상
    if (b.epSquare != -1) {
      for (int df = -1; df <= 1; df += 2) {
        int cf = file + df;
        if (cf < 0 || cf >= 8)
          continue;
        int cr = nextRank;
        int to = cr * 8 + cf;
        if (to == b.epSquare) {
          addMove(out, sq, to);
        }
      }
    }
  }
}

// 나이트 pseudo-legal
static void genKnightMoves(const Board &b, int sq, std::vector<Move> &out) {
  Piece p = b.pieceAt(sq);
  if (p == NO_PIECE)
    return;
  Color us = b.stm;
  if (isWhite(p) != (us == WHITE))
    return;
  if (pieceType(p) != KNIGHT)
    return;

  static const int knightOffsets[8][2] = {{1, 2},   {2, 1},   {2, -1}, {1, -2},
                                          {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};

  int file = sq % 8;
  int rank = sq / 8;

  for (auto &o : knightOffsets) {
    int rf = rank + o[1];
    int ff = file + o[0];
    if (rf < 0 || rf >= 8 || ff < 0 || ff >= 8)
      continue;
    int to = rf * 8 + ff;
    Piece target = b.pieceAt(to);
    if (target == NO_PIECE || isWhite(target) != (us == WHITE)) {
      addMove(out, sq, to);
    }
  }
}

// 슬라이딩 말 (비숍/룩/퀸) 공통
static void genSlidingMoves(const Board &b, int sq, std::vector<Move> &out,
                            const int dirs[][2], int dirCount) {
  Piece p = b.pieceAt(sq);
  if (p == NO_PIECE)
    return;
  Color us = b.stm;
  if (isWhite(p) != (us == WHITE))
    return;

  int file = sq % 8;
  int rank = sq / 8;

  for (int i = 0; i < dirCount; ++i) {
    int df = dirs[i][0];
    int dr = dirs[i][1];
    int rf = rank + dr;
    int ff = file + df;
    while (rf >= 0 && rf < 8 && ff >= 0 && ff < 8) {
      int to = rf * 8 + ff;
      Piece target = b.pieceAt(to);
      if (target == NO_PIECE) {
        addMove(out, sq, to);
      } else {
        if (isWhite(target) != (us == WHITE)) {
          addMove(out, sq, to);
        }
        break;
      }
      rf += dr;
      ff += df;
    }
  }
}

// 킹 pseudo-legal (캐슬 포함)
static void genKingMoves(const Board &b, int sq, std::vector<Move> &out) {
  Piece p = b.pieceAt(sq);
  if (p == NO_PIECE)
    return;
  Color us = b.stm;
  if (isWhite(p) != (us == WHITE))
    return;
  if (pieceType(p) != KING)
    return;

  int file = sq % 8;
  int rank = sq / 8;

  // 일반 한 칸 이동
  for (int dr = -1; dr <= 1; ++dr) {
    for (int df = -1; df <= 1; ++df) {
      if (dr == 0 && df == 0)
        continue;
      int rf = rank + dr;
      int ff = file + df;
      if (rf < 0 || rf >= 8 || ff < 0 || ff >= 8)
        continue;
      int to = rf * 8 + ff;
      Piece target = b.pieceAt(to);
      if (target == NO_PIECE || isWhite(target) != (us == WHITE)) {
        addMove(out, sq, to);
      }
    }
  }

  // 캐슬링 (pseudo-legal, 공격 여부는 나중에 필터)
  if (us == WHITE && rank == 0 && file == 4) {
    // king side: e1(4) -> g1(6)
    if (b.castleWK && b.pieceAt(5) == NO_PIECE && b.pieceAt(6) == NO_PIECE) {
      addMove(out, sq, 6);
    }
    // queen side: e1(4) -> c1(2)
    if (b.castleWQ && b.pieceAt(3) == NO_PIECE && b.pieceAt(2) == NO_PIECE &&
        b.pieceAt(1) == NO_PIECE) {
      addMove(out, sq, 2);
    }
  }
  if (us == BLACK && rank == 7 && file == 4) {
    // king side: e8(60) -> g8(62)
    if (b.castleBK && b.pieceAt(61) == NO_PIECE && b.pieceAt(62) == NO_PIECE) {
      addMove(out, sq, 62);
    }
    // queen side: e8(60) -> c8(58)
    if (b.castleBQ && b.pieceAt(59) == NO_PIECE && b.pieceAt(58) == NO_PIECE &&
        b.pieceAt(57) == NO_PIECE) {
      addMove(out, sq, 58);
    }
  }
}

// pseudo-legal 전체 생성
static void generatePseudo(const Board &b, std::vector<Move> &out) {
  out.clear();

  Color us = b.stm;

  // 폰
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = b.pieceAt(sq);
    if (p == NO_PIECE)
      continue;
    if (pieceType(p) == PAWN && isWhite(p) == (us == WHITE)) {
      genPawnMoves(b, sq, out);
    }
  }

  // 나이트
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = b.pieceAt(sq);
    if (p == NO_PIECE)
      continue;
    if (pieceType(p) == KNIGHT && isWhite(p) == (us == WHITE)) {
      genKnightMoves(b, sq, out);
    }
  }

  // 비숍/퀸 (대각선)
  {
    static const int bishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int sq = 0; sq < 64; ++sq) {
      Piece p = b.pieceAt(sq);
      if (p == NO_PIECE)
        continue;
      if (isWhite(p) != (us == WHITE))
        continue;
      PieceType pt = pieceType(p);
      if (pt == BISHOP || pt == QUEEN) {
        genSlidingMoves(b, sq, out, bishopDirs, 4);
      }
    }
  }

  // 룩/퀸 (직선)
  {
    static const int rookDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int sq = 0; sq < 64; ++sq) {
      Piece p = b.pieceAt(sq);
      if (p == NO_PIECE)
        continue;
      if (isWhite(p) != (us == WHITE))
        continue;
      PieceType pt = pieceType(p);
      if (pt == ROOK || pt == QUEEN) {
        genSlidingMoves(b, sq, out, rookDirs, 4);
      }
    }
  }

  // 킹
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = b.pieceAt(sq);
    if (p == NO_PIECE)
      continue;
    if (pieceType(p) == KING && isWhite(p) == (us == WHITE)) {
      genKingMoves(b, sq, out);
    }
  }
}

// legal 필터
void MoveGen::generate(const Board &b, std::vector<Move> &out) {
  std::vector<Move> pseudo;
  generatePseudo(b, pseudo);

  out.clear();
  out.reserve(pseudo.size());

  Color us = b.stm;

  for (const Move &m : pseudo) {
    Board tmp = b;
    tmp.makeMove(m);

    // 우리 킹 위치 찾기
    int kingSq = -1;
    for (int sq = 0; sq < 64; ++sq) {
      Piece p = tmp.pieceAt(sq);
      if (p != NO_PIECE && pieceType(p) == KING &&
          isWhite(p) == (us == WHITE)) {
        kingSq = sq;
        break;
      }
    }
    if (kingSq == -1) {
      continue;
    }

    // 상대가 우리 킹을 공격하면 illegal
    if (!tmp.isSquareAttacked(kingSq, opposite(us))) {
      out.push_back(m);
    }
  }
}

} // namespace chess
