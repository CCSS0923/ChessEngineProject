#pragma once
#include <vector>
#include "move.h"
#include "board.h"
namespace chess { struct MoveGen{ static void generate(const Board&, std::vector<Move>&); }; }
