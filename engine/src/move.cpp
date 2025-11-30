#include "move.h"

namespace chess {

static std::string squareToUci(int sq) {
  int file = sq % 8;
  int rank = sq / 8;
  std::string s;
  s.push_back(static_cast<char>('a' + file));
  s.push_back(static_cast<char>('1' + rank));
  return s;
}

static char promoChar(Piece p) {
  PieceType pt = pieceType(p);
  switch (pt) {
  case QUEEN:
    return 'q';
  case ROOK:
    return 'r';
  case BISHOP:
    return 'b';
  case KNIGHT:
    return 'n';
  default:
    return 'q';
  }
}

std::string Move::toUciString() const {
  std::string s = squareToUci(from) + squareToUci(to);
  if (promotion != NO_PIECE) {
    s.push_back(promoChar(promotion));
  }
  return s;
}

} // namespace chess
