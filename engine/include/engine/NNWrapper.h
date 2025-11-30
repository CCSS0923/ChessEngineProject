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

  // 기존: tensor 입력 버전
  NNResult evaluate(const uint8_t *inputCHW);

  // 새로 추가: Board → policy vector
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
};

} // namespace chess
