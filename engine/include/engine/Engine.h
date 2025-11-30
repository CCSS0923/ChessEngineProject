#pragma once

#include <string>

#include "board.h"
#include "move.h"

namespace chess {

struct NNWrapper; // 전방 선언

struct Engine {
  Board b;

  void NewGame();
  void SetPositionFromUCI(const std::string &cmd);

  // UCI "go ..." 명령 문자열까지 받아서 처리
  std::string SearchBestMove(NNWrapper &nn, const std::string &goCmd);
};

} // namespace chess
