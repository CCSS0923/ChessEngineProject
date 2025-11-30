#pragma once

#include <array>
#include <mutex>
#include <string>
#include <vector>

#include "NNBoardTensor.h" // boardToTensor()
#include "board.h"         // Board 클래스
#include <torch/script.h>

namespace chess {

struct NNResult {
  std::array<float, 4096> policy{};
  float value = 0.0f;
  bool ok = false;
};

class NNWrapper {
public:
  static NNWrapper &instance();

  void init(const std::string &modelPath, int deviceIndex = -1);
  bool isReady() const;

  // 치명적인 오류 이후 NN 사용 비활성화
  void disable();

  // 마지막 오류 메시지 조회
  const std::string &lastError() const;

  // 기존: tensor 입력 버전
  NNResult evaluate(const uint8_t *inputCHW);

  // Board → policy vector
  std::vector<float> Evaluate(const Board &board);

private:
  NNWrapper();
  ~NNWrapper() = default;

  NNWrapper(const NNWrapper &) = delete;
  NNWrapper &operator=(const NNWrapper &) = delete;

  mutable std::mutex m_mutex;
  torch::jit::script::Module m_module;
  torch::Device m_device;
  bool m_ready = false;
  std::string m_lastError;
};

} // namespace chess
