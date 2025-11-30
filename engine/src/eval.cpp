#include "eval.h"
#include "NNBoardTensor.h"
#include "NNWrapper.h"
#include "types.h"

namespace chess {

// ------------------------------------------
// Classical fallback evaluation (optional)
// 필요하면 나중에 material / PST 추가 가능
// ------------------------------------------
static int classicalEvaluate(const Board &b) {
  // 최소 더미 평가 (0점)
  // 필요하면 네가 원래 쓰던 value 계산 넣어라.
  return 0;
}

// ------------------------------------------
// NN 기반 evaluate (화이트 기준)
// ------------------------------------------
int evaluate(const Board &b) {
  NNWrapper &nn = NNWrapper::instance();

  // NN 로딩 안 되었으면 classical
  if (!nn.isReady()) {
    return classicalEvaluate(b);
  }

  // 텐서 생성
  uint8_t tensor[18 * 8 * 8];
  boardToTensor(b, tensor);

  // NN inference
  NNResult r = nn.evaluate(tensor);

  // NN 실패 → classical fallback
  if (!r.ok) {
    return classicalEvaluate(b);
  }

  // NN value: [-1,1], side-to-move 기준이 아니라 "화이트 관점"
  // search.cpp에서 side-to-move 부호는 negamax 내부에서 조정한다.
  float v = r.value;

  // centipawn 변환 (±1000 정도 추천)
  int score = static_cast<int>(v * 1000.0f);
  return score;
}

} // namespace chess
