#include "board.h"
#include "move.h"
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace chess {

static Piece charToPiece(char c) {
  switch (c) {
  case 'P':
    return WP;
  case 'N':
    return WN;
  case 'B':
    return WB;
  case 'R':
    return WR;
  case 'Q':
    return WQ;
  case 'K':
    return WK;
  case 'p':
    return BP;
  case 'n':
    return BN;
  case 'b':
    return BB;
  case 'r':
    return BR;
  case 'q':
    return BQ;
  case 'k':
    return BK;
  default:
    return NO_PIECE;
  }
}

bool Board::loadFEN(const std::string &fen) {
  // 초기화
  for (int i = 0; i < 64; ++i)
    squares[i] = NO_PIECE;

  stm = WHITE;
  castleWK = castleWQ = castleBK = castleBQ = false;
  epSquare = -1;

  std::istringstream ss(fen);
  std::string placement, stmField, castlingField, epField;
  std::string halfmove, fullmove; // 사용 안 해도 일단 읽기

  if (!(ss >> placement))
    return false;
  if (!(ss >> stmField))
    stmField = "w";
  if (!(ss >> castlingField))
    castlingField = "-";
  if (!(ss >> epField))
    epField = "-";
  ss >> halfmove >> fullmove; // 안 쓰지만 읽어둠

  int rank = 7; // 8번째 rank부터 내려옴
  int file = 0;

  for (char c : placement) {
    if (c == '/') {
      rank--;
      file = 0;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) {
      file += c - '0';
      continue;
    }
    Piece p = charToPiece(c);
    if (p == NO_PIECE)
      return false;

    if (file < 0 || file > 7 || rank < 0 || rank > 7)
      return false;

    int sq = rank * 8 + file;
    squares[sq] = p;
    file++;
  }

  if (stmField == "w")
    stm = WHITE;
  else if (stmField == "b")
    stm = BLACK;
  else
    stm = WHITE;

  if (castlingField != "-") {
    for (char c : castlingField) {
      switch (c) {
      case 'K':
        castleWK = true;
        break;
      case 'Q':
        castleWQ = true;
        break;
      case 'k':
        castleBK = true;
        break;
      case 'q':
        castleBQ = true;
        break;
      default:
        break;
      }
    }
  }

  if (epField != "-") {
    if (epField.size() == 2) {
      int fileEp = epField[0] - 'a';
      int rankEp = epField[1] - '1';
      if (fileEp >= 0 && fileEp < 8 && rankEp >= 0 && rankEp < 8) {
        epSquare = rankEp * 8 + fileEp;
      }
    }
  }

  return true;
}

Piece Board::pieceAt(int idx) const {
  if (idx < 0 || idx >= 64)
    return NO_PIECE;
  return squares[idx];
}

// 공격 여부 판정
bool Board::isSquareAttacked(int sq, Color by) const {
  int file = sq % 8;
  int rank = sq / 8;

  // 폰
  {
    int dir = (by == WHITE) ? -1 : 1; // 공격 방향은 상대 기준
    int r = rank + dir;
    if (r >= 0 && r < 8) {
      int f1 = file - 1;
      int f2 = file + 1;

      if (f1 >= 0) {
        int s = r * 8 + f1;
        Piece p = pieceAt(s);
        if (p != NO_PIECE && isWhite(p) == (by == WHITE) &&
            pieceType(p) == PAWN)
          return true;
      }
      if (f2 < 8) {
        int s = r * 8 + f2;
        Piece p = pieceAt(s);
        if (p != NO_PIECE && isWhite(p) == (by == WHITE) &&
            pieceType(p) == PAWN)
          return true;
      }
    }
  }

  // 나이트
  {
    static const int knightOffsets[8][2] = {
        {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    for (auto &o : knightOffsets) {
      int rf = rank + o[1];
      int ff = file + o[0];
      if (rf < 0 || rf >= 8 || ff < 0 || ff >= 8)
        continue;
      int s = rf * 8 + ff;
      Piece p = pieceAt(s);
      if (p != NO_PIECE && isWhite(p) == (by == WHITE) &&
          pieceType(p) == KNIGHT)
        return true;
    }
  }

  // 비숍/퀸 (대각선)
  {
    static const int dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (auto &d : dirs) {
      int rf = rank + d[1];
      int ff = file + d[0];
      while (rf >= 0 && rf < 8 && ff >= 0 && ff < 8) {
        int s = rf * 8 + ff;
        Piece p = pieceAt(s);
        if (p != NO_PIECE) {
          if (isWhite(p) == (by == WHITE)) {
            PieceType pt = pieceType(p);
            if (pt == BISHOP || pt == QUEEN)
              return true;
          }
          break;
        }
        rf += d[1];
        ff += d[0];
      }
    }
  }

  // 룩/퀸 (직선)
  {
    static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (auto &d : dirs) {
      int rf = rank + d[1];
      int ff = file + d[0];
      while (rf >= 0 && rf < 8 && ff >= 0 && ff < 8) {
        int s = rf * 8 + ff;
        Piece p = pieceAt(s);
        if (p != NO_PIECE) {
          if (isWhite(p) == (by == WHITE)) {
            PieceType pt = pieceType(p);
            if (pt == ROOK || pt == QUEEN)
              return true;
          }
          break;
        }
        rf += d[1];
        ff += d[0];
      }
    }
  }

  // 킹
  {
    for (int dr = -1; dr <= 1; ++dr) {
      for (int df = -1; df <= 1; ++df) {
        if (dr == 0 && df == 0)
          continue;
        int rf = rank + dr;
        int ff = file + df;
        if (rf < 0 || rf >= 8 || ff < 0 || ff >= 8)
          continue;
        int s = rf * 8 + ff;
        Piece p = pieceAt(s);
        if (p != NO_PIECE && isWhite(p) == (by == WHITE) &&
            pieceType(p) == KING)
          return true;
      }
    }
  }

  return false;
}

// 실제 말 움직이기 (프로모션, 앙파상, 캐슬링 포함)
void Board::makeMove(const Move &m) {
  Piece moving = squares[m.from];
  PieceType mt = pieceType(moving);
  Color us = stm;

  int fromFile = m.from % 8;
  int fromRank = m.from / 8;
  int toFile = m.to % 8;
  int toRank = m.to / 8;

  // 기본: 캡처
  Piece captured = squares[m.to];

  // 앙파상인지 판별 (폰이 epSquare로 이동하고, 도착 칸은 비어있어야 함)
  bool isEP = false;
  if (mt == PAWN && m.to == epSquare && captured == NO_PIECE) {
    isEP = true;
    int capRank = (us == WHITE) ? toRank - 1 : toRank + 1;
    int capSq = capRank * 8 + toFile;
    captured = squares[capSq];
    squares[capSq] = NO_PIECE;
  }

  // 캐슬링인지 판별 (킹이 두 칸 이동)
  bool isCastle = (mt == KING && std::abs(toFile - fromFile) == 2);
  if (isCastle) {
    if (us == WHITE) {
      // e1(4) 기준: g1(6) → king side, c1(2) → queen side
      if (toFile == 6) {
        // king side: h1(7) → f1(5)
        squares[5] = squares[7];
        squares[7] = NO_PIECE;
      } else if (toFile == 2) {
        // queen side: a1(0) → d1(3)
        squares[3] = squares[0];
        squares[0] = NO_PIECE;
      }
    } else {
      // e8(60) 기준: g8(62), c8(58)
      if (toFile == 6) {
        // king side: h8(63) → f8(61)
        squares[61] = squares[63];
        squares[63] = NO_PIECE;
      } else if (toFile == 2) {
        // queen side: a8(56) → d8(59)
        squares[59] = squares[56];
        squares[56] = NO_PIECE;
      }
    }
  }

  // 말 이동
  squares[m.from] = NO_PIECE;
  Piece placed = moving;

  // 프로모션
  if (mt == PAWN && (toRank == 0 || toRank == 7) && m.promotion != NO_PIECE) {
    placed = m.promotion;
  }
  squares[m.to] = placed;

  // 캐슬링 권리 갱신
  if (moving == WK) {
    castleWK = castleWQ = false;
  } else if (moving == BK) {
    castleBK = castleBQ = false;
  }

  if (moving == WR && m.from == 0)
    castleWQ = false;
  if (moving == WR && m.from == 7)
    castleWK = false;
  if (moving == BR && m.from == 56)
    castleBQ = false;
  if (moving == BR && m.from == 63)
    castleBK = false;

  if (captured == WR && m.to == 0)
    castleWQ = false;
  if (captured == WR && m.to == 7)
    castleWK = false;
  if (captured == BR && m.to == 56)
    castleBQ = false;
  if (captured == BR && m.to == 63)
    castleBK = false;

  // 새로운 epSquare 설정
  if (mt == PAWN && std::abs(toRank - fromRank) == 2) {
    int midRank = (toRank + fromRank) / 2;
    epSquare = midRank * 8 + fromFile;
  } else {
    epSquare = -1;
  }

  // 턴 교체
  stm = opposite(stm);
}

} // namespace chess
