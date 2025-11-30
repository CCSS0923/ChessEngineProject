#include "engine/NNWrapper.h"
#include "engine/NNBoardTensor.h"
#include "engine/board.h"

#include <cstring>
#include <fstream>
#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

namespace chess {

NNWrapper &NNWrapper::instance() {
  static NNWrapper inst;
  return inst;
}

NNWrapper::NNWrapper() : m_device(torch::kCPU), m_ready(false) {}

bool NNWrapper::isReady() const { return m_ready; }

void NNWrapper::init(const std::string &modelPath, int deviceIndex) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_ready = false;

  // 1) 먼저 파일 존재 / 열기 확인
  std::ifstream ifs(modelPath, std::ios::binary);
  if (!ifs) {
    std::cerr << "[NNWrapper] model not found: " << modelPath << std::endl;
    return;
  }

  try {
    // 2) 디바이스 결정
    if (deviceIndex >= 0 && torch::cuda::is_available()) {
      m_device = torch::Device(torch::kCUDA, deviceIndex);
      std::cout << "[NNWrapper] Using CUDA:" << deviceIndex << "\n";
    } else {
      m_device = torch::Device(torch::kCPU);
      std::cout << "[NNWrapper] Using CPU\n";
    }

    // 3) 경로 기반 오버로드 대신, 스트림 기반 오버로드 사용
    m_module = torch::jit::load(ifs, m_device);
    m_module.eval();
    m_ready = true;

    std::cout << "[NNWrapper] Model loaded OK\n";
  } catch (const c10::Error &e) {
    std::cerr << "[NNWrapper] load error: " << e.what() << std::endl;
    m_ready = false;
  }
}

NNResult NNWrapper::evaluate(const uint8_t *inputCHW) {
  NNResult result;
  if (!m_ready)
    return result;

  try {
    auto options_u8 =
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    torch::Tensor input_u8 = torch::from_blob(const_cast<uint8_t *>(inputCHW),
                                              {1, 18, 8, 8}, options_u8)
                                 .clone();

    torch::Tensor input =
        input_u8.to(torch::kFloat32).div_(255.0f).to(m_device);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input);

    torch::jit::IValue out = m_module.forward(inputs);
    auto tup = out.toTuple();

    auto policy = tup->elements()[0].toTensor().to(torch::kCPU).view({4096});
    auto value = tup->elements()[1].toTensor().to(torch::kCPU).view({1});

    for (int i = 0; i < 4096; ++i)
      result.policy[i] = policy[i].item<float>();

    result.value = value[0].item<float>();
    result.ok = true;
  } catch (...) {
    std::cerr << "[NNWrapper] evaluate failed\n";
  }

  return result;
}

// ★★★★★ 핵심: Board → policy vector 버전
std::vector<float> NNWrapper::Evaluate(const Board &board) {
  std::vector<float> out(4096, 0.0f);

  if (!m_ready)
    return out;

  uint8_t tensor[18 * 8 * 8];
  boardToTensor(board, tensor);

  NNResult r = evaluate(tensor);
  if (!r.ok)
    return out;

  for (int i = 0; i < 4096; ++i)
    out[i] = r.policy[i];

  return out;
}

} // namespace chess
