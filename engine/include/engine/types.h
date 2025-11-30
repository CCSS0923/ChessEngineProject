#pragma once

namespace chess {

using Bitboard = unsigned long long;

enum Color { WHITE, BLACK };

inline Color opposite(Color c) { return c == WHITE ? BLACK : WHITE; }

enum PieceType { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NONE_TYPE };

// 기존 enum 순서를 그대로 유지 (기존 코드와의 호환성)
enum Piece { WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, NO_PIECE };

inline bool isWhite(Piece p) { return (p >= WP && p <= WK); }

inline bool isBlack(Piece p) { return (p >= BP && p <= BK); }

inline PieceType pieceType(Piece p) {
  switch (p) {
  case WP:
  case BP:
    return PAWN;
  case WN:
  case BN:
    return KNIGHT;
  case WB:
  case BB:
    return BISHOP;
  case WR:
  case BR:
    return ROOK;
  case WQ:
  case BQ:
    return QUEEN;
  case WK:
  case BK:
    return KING;
  default:
    return NONE_TYPE;
  }
}

} // namespace chess
