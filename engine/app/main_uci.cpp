#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Engine.h"
#include "NNWrapper.h"

using namespace chess;

// --------------------------------------
// 간단한 문자열 분리 유틸
// --------------------------------------
static std::vector<std::string> split(const std::string &s) {
  std::stringstream ss(s);
  std::vector<std::string> out;
  std::string tok;
  while (ss >> tok)
    out.push_back(tok);
  return out;
}

// --------------------------------------
// UCI 옵션용 전역 상태
// --------------------------------------
static std::string g_nnModelFile = "../checkpoints/chessnet_ts.pt";
static std::string g_nnDevice = "cpu"; // "cpu", "cuda0", "cuda1", ...

// setoption 파싱
static void handleSetOption(const std::string &line) {
  // 예:
  // setoption name NNModelFile value ../checkpoints/chessnet_ts.pt
  // setoption name NNDevice    value cuda0
  std::istringstream iss(line);
  std::string tok;
  std::string name;
  std::string value;

  iss >> tok; // setoption
  if (!(iss >> tok))
    return; // name
  if (!(iss >> name))
    return; // 실제 option 이름

  // 남은 부분에서 "value" 뒤를 value로 취급
  while (iss >> tok) {
    if (tok == "value")
      break;
  }
  std::getline(iss, value);
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
    value.erase(value.begin());

  if (name == "NNModelFile") {
    g_nnModelFile = value;
  } else if (name == "NNDevice") {
    g_nnDevice = value;
  }
}

int main() {
  Engine engine;

  std::string line;

  while (std::getline(std::cin, line)) {
    auto tok = split(line);
    if (tok.empty())
      continue;

    const std::string &cmd = tok[0];

    if (cmd == "uci") {
      std::cout << "id name MyNeuralChessEngine\n";
      std::cout << "id author You\n";
      // NN 관련 UCI 옵션 알림
      std::cout << "option name NNModelFile type string default "
                   "../checkpoints/chessnet_ts.pt\n";
      std::cout << "option name NNDevice type string default cpu\n";
      std::cout << "uciok\n";
    } else if (cmd == "setoption") {
      handleSetOption(line);
    } else if (cmd == "isready") {
      // UCI GUI가 isready를 보낼 때, 이 시점에 NN 로드
      int deviceIndex = -1;                   // -1 => CPU
      if (g_nnDevice.rfind("cuda", 0) == 0) { // "cuda0", "cuda1", ...
        try {
          deviceIndex = std::stoi(g_nnDevice.substr(4));
        } catch (...) {
          deviceIndex = 0;
        }
      }

      NNWrapper::instance().init(g_nnModelFile, deviceIndex);
      std::cout << "readyok\n";
    } else if (cmd == "ucinewgame") {
      engine.NewGame();
    } else if (cmd == "position") {
      // line 전체를 Engine 쪽으로 넘기면,
      // Engine::SetPositionFromUCI가 "startpos" / "fen" 기준으로 처리
      engine.SetPositionFromUCI(line);
    } else if (cmd == "go") {
      // go depth / movetime 등 전체 문자열을 그대로 넘김
      // Engine::SearchBestMove에서 depth 파싱 (parseDepthFromGo) 수행
      NNWrapper &nn = NNWrapper::instance();
      std::string bestMove = engine.SearchBestMove(nn, line);
      std::cout << "bestmove " << bestMove << "\n";
    } else if (cmd == "quit") {
      break;
    }
  }

  return 0;
}
