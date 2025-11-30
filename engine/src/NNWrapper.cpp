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

void NNWrapper::disable() {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_ready = false;
}

const std::string &NNWrapper::lastError() const { return m_lastError; }

void NNWrapper::init(const std::string &modelPath, int deviceIndex) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_ready = false;
  m_lastError.clear();

  std::ifstream ifs(modelPath, std::ios::binary);
  if (!ifs) {
    m_lastError = "model not found: " + modelPath;
    std::cerr << "[NNWrapper] " << m_lastError << std::endl;
    return;
  }

  try {
    // 1) 디바이스 결정
    if (deviceIndex >= 0 && torch::cuda::is_available()) {
      m_device = torch::Device(torch::kCUDA, deviceIndex);
      std::cout << "[NNWrapper] Using CUDA:" << deviceIndex << "\n";
    } else {
      m_device = torch::Device(torch::kCPU);
      std::cout << "[NNWrapper] Using CPU\n";
    }

    // 2) 모델 로드
    m_module = torch::jit::load(ifs, m_device);
    m_module.eval();
  } catch (const c10::Error &e) {
    m_lastError = e.what_without_backtrace();
    std::cerr << "[NNWrapper] load error(c10): " << m_lastError << std::endl;
    return;
  } catch (const std::exception &e) {
    m_lastError = e.what();
    std::cerr << "[NNWrapper] load error(std): " << m_lastError << std::endl;
    return;
  } catch (...) {
    m_lastError = "unknown exception in init()";
    std::cerr << "[NNWrapper] load error: " << m_lastError << std::endl;
    return;
  }

  // 3) forward 검증
  try {
    torch::Tensor dummy = torch::zeros(
        {1, 18, 8, 8},
        torch::TensorOptions().dtype(torch::kFloat32).device(m_device));

    std::vector<torch::jit::IValue> inp;
    inp.emplace_back(dummy);

    auto out = m_module.forward(inp);
    auto tup = out.toTuple();

    if (!tup || tup->elements().size() < 2 || !tup->elements()[0].isTensor() ||
        !tup->elements()[1].isTensor()) {
      m_lastError = "model forward validation failed";
      std::cerr << "[NNWrapper] " << m_lastError << "\n";
      return;
    }

  } catch (const c10::Error &e) {
    m_lastError =
        std::string("forward validation c10: ") + e.what_without_backtrace();
    std::cerr << "[NNWrapper] " << m_lastError << "\n";
    return;
  } catch (const std::exception &e) {
    m_lastError = std::string("forward validation std: ") + e.what();
    std::cerr << "[NNWrapper] " << m_lastError << "\n";
    return;
  } catch (...) {
    m_lastError = "model forward validation threw unknown exception";
    std::cerr << "[NNWrapper] " << m_lastError << "\n";
    return;
  }

  // 4) 여기까지 왔으면 진짜로 ready
  m_ready = true;
  m_lastError.clear();
  std::cout << "[NNWrapper] Model loaded OK\n";
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

    if (!out.isTuple()) {
      m_lastError = "model output is not tuple";
      std::cerr << "[NNWrapper] " << m_lastError << "\n";
      m_ready = false;
      return result;
    }

    auto tup = out.toTuple();
    if (!tup || tup->elements().size() < 2) {
      m_lastError = "model output tuple size < 2";
      std::cerr << "[NNWrapper] " << m_lastError << "\n";
      m_ready = false;
      return result;
    }

    auto policyI = tup->elements()[0];
    auto valueI = tup->elements()[1];

    if (!policyI.isTensor() || !valueI.isTensor()) {
      m_lastError = "model output elements are not tensors";
      std::cerr << "[NNWrapper] " << m_lastError << "\n";
      m_ready = false;
      return result;
    }

    auto policyT = policyI.toTensor().to(torch::kCPU);
    auto valueT = valueI.toTensor().to(torch::kCPU);

    if (!policyT.defined() || policyT.numel() != 4096) {
      m_lastError = "policy tensor has invalid shape, numel=" +
                    std::to_string(policyT.numel());
      std::cerr << "[NNWrapper] " << m_lastError << "\n";
      m_ready = false;
      return result;
    }

    policyT = policyT.view({4096});
    valueT = valueT.view(-1);

    if (!valueT.defined() || valueT.numel() < 1) {
      m_lastError = "value tensor has invalid shape";
      std::cerr << "[NNWrapper] " << m_lastError << "\n";
      m_ready = false;
      return result;
    }

    for (int i = 0; i < 4096; ++i)
      result.policy[i] = policyT[i].item<float>();

    result.value = valueT[0].item<float>();
    result.ok = true;
    m_lastError.clear();
  } catch (const c10::Error &e) {
    m_lastError = e.what_without_backtrace();
    std::cerr << "[NNWrapper] evaluate failed(c10): " << m_lastError << "\n";
    m_ready = false;
  } catch (const std::exception &e) {
    m_lastError = e.what();
    std::cerr << "[NNWrapper] evaluate failed(std): " << m_lastError << "\n";
    m_ready = false;
  } catch (...) {
    m_lastError = "unknown exception in evaluate()";
    std::cerr << "[NNWrapper] evaluate failed(unknown)\n";
    m_ready = false;
  }

  return result;
}

// Board → policy vector 버전
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
