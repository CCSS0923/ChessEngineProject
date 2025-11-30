#include "Engine.h"
#include "NNWrapper.h"
#include "movegen.h"

#include <sstream>

namespace chess {

namespace {

const char *START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

int parseDepthFromGo(const std::string &goCmd, int defaultDepth = 4) {
  std::istringstream iss(goCmd);
  std::string tok;
  while (iss >> tok) {
    if (tok == "depth") {
      int d;
      if (iss >> d)
        return d;
    }
  }
  return defaultDepth;
}

} // namespace

void Engine::NewGame() { b.loadFEN(START_FEN); }

void Engine::SetPositionFromUCI(const std::string &cmd) {
  if (cmd.find("startpos") != std::string::npos) {
    b.loadFEN(START_FEN);
    return;
  }

  std::size_t fenPos = cmd.find("fen");
  if (fenPos != std::string::npos) {
    std::string fenPart = cmd.substr(fenPos + 3);
    std::size_t movesPos = fenPart.find(" moves");
    if (movesPos != std::string::npos)
      fenPart = fenPart.substr(0, movesPos);

    while (!fenPart.empty() &&
           (fenPart.front() == ' ' || fenPart.front() == '\t'))
      fenPart.erase(fenPart.begin());
    while (!fenPart.empty() &&
           (fenPart.back() == ' ' || fenPart.back() == '\t'))
      fenPart.pop_back();

    if (!fenPart.empty())
      b.loadFEN(fenPart);
  }
}

std::string Engine::SearchBestMove(NNWrapper &nn, const std::string &goCmd) {
  int depth = parseDepthFromGo(goCmd, 4);
  (void)depth;

  std::vector<Move> moves;
  MoveGen::generate(b, moves);

  if (moves.empty())
    return "0000";

  // ★★★ 수정 후 정상 작동: Board → NNWrapper.Evaluate(Board)
  std::vector<float> policy = nn.Evaluate(b);

  float bestScore = -1e30f;
  Move best = moves.front();

  for (const auto &m : moves) {
    if (m.from < 0 || m.from >= 64 || m.to < 0 || m.to >= 64)
      continue;

    float s = policy[m.from * 64 + m.to];
    if (s > bestScore) {
      bestScore = s;
      best = m;
    }
  }

  return best.toUciString();
}

} // namespace chess
