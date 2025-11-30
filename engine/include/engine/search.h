#pragma once

#include "board.h"
#include "move.h"
#include <vector>

namespace chess {

// negamax는 내부 static 함수이므로 search.h에는 선언하지 않는다.

// searchBest:
//   - 현재 board 상태에서 depth 만큼 탐색해 최선의 Move 반환
//   - side-to-move 기준
Move searchBest(Board &b, int depth);

} // namespace chess
