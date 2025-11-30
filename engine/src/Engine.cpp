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
  // go 명령에서 depth 파싱 (지금은 아직 사용 안 하지만, 나중 search.cpp 연동용)
  int depth = parseDepthFromGo(goCmd, 4);
  (void)depth;

  // 1) 합법 수 생성
  std::vector<Move> moves;
  MoveGen::generate(b, moves);

  if (moves.empty())
    return "0000";

  // 2) NN 사용 가능한지 체크
  bool useNN = nn.isReady();
  std::vector<float> policy;

  if (useNN) {
    policy = nn.Evaluate(b);

    // 정책 벡터 크기 검증
    if (policy.size() != 4096) {
      std::cerr << "[Engine] NN policy size invalid: " << policy.size()
                << " (expected 4096)\n";
      useNN = false;
    } else {
      // 전부 0이면 NN 실패로 간주
      bool allZero = true;
      for (float v : policy) {
        if (v != 0.0f) {
          allZero = false;
          break;
        }
      }
      if (allZero) {
        std::cerr << "[Engine] NN policy all zero. Fallback to simple move\n";
        useNN = false;
      }
    }
  }

  Move best = moves.front();

  // 3) NN 정책을 사용해서 bestmove 선택
  if (useNN) {
    float bestScore = -1e30f;
    bool found = false;

    for (const auto &m : moves) {
      // from/to 범위 체크
      if (m.from < 0 || m.from >= 64 || m.to < 0 || m.to >= 64)
        continue;

      int idx = m.from * 64 + m.to;
      if (idx < 0 || idx >= static_cast<int>(policy.size()))
        continue;

      float s = policy[static_cast<std::size_t>(idx)];
      if (!found || s > bestScore) {
        bestScore = s;
        best = m;
        found = true;
      }
    }

    // NN 정책으로 선택된 수가 있으면 바로 반환
    if (found)
      return best.toUciString();
  }

  // 4) NN이 비정상(ready 아님 / 출력 이상 / all zero)일 경우
  //    → 일단 첫 legal move로 폴백
  return best.toUciString();
}
} // namespace chess
