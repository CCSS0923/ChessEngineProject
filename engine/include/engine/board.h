#pragma once
#include "types.h"
#include <string>

namespace chess {

struct Move; // 전방 선언

struct Board {
  // 0 = a1, 1 = b1, ... 7 = h1, 8 = a2, ... 63 = h8
  Piece squares[64]{};
  Color stm = WHITE;

  bool castleWK = false; // White king side
  bool castleWQ = false; // White queen side
  bool castleBK = false; // Black king side
  bool castleBQ = false; // Black queen side

  int epSquare = -1; // 앙파상 가능 칸(도착 칸), 없으면 -1

  bool loadFEN(const std::string &fen);
  Piece pieceAt(int idx) const;

  void makeMove(const Move &m);
  bool isSquareAttacked(int sq, Color by) const;
};

} // namespace chess
