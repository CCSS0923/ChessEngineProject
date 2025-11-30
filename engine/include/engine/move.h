#pragma once
#include "types.h"
#include <string>

namespace chess {

struct Move {
  int from;
  int to;
  Piece promotion; // NO_PIECE 아니면 프로모션

  Move(int f = 0, int t = 0, Piece promo = NO_PIECE)
      : from(f), to(t), promotion(promo) {}

  std::string toUciString() const;
};

} // namespace chess
