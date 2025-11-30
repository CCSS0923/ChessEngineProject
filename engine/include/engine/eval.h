#pragma once

#include "board.h"

namespace chess {

// NN value 기반 평가 함수 (화이트 관점 기준)
int evaluate(const Board &b);

} // namespace chess
