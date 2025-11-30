#pragma once
#include "board.h"
#include <cstdint>

namespace chess {

// outCHW: 18 * 8 * 8 = [channel][rank][file], CHW 순서, uint8
void boardToTensor(const Board &board, uint8_t outCHW[18 * 8 * 8]);

} // namespace chess
