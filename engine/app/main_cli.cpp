#include <iostream>
#include <string>
#include "chess/board.h"
#include "chess/search.h"
#include "chess/movegen.h"
#include "chess/uci.h"

using namespace chess;

int main() {
    Board board = Board::fromFEN(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    );

    SearchParams params;
    params.maxDepth = 2;
    Search search(params);

    while (true) {
        std::cout << "Position: " << board.toFEN() << "\n";
        std::cout << "Your move (uci, 'q' to quit): ";

        std::string uci;
        if (!(std::cin >> uci)) break;
        if (uci == "q") break;

        Move userMove = uciToMove(board, uci);

        if (!userMove.isValid()) {
            std::cout << "Illegal move!\n";
            continue;
        }

        board.makeMove(userMove);

        Move best = search.findBestMove(board);
        std::cout << "Engine played: " << best.toString() << "\n";
        board.makeMove(best);
    }

    return 0;
}
