#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "training/dataset.h"
#include "training/model.h"

namespace fs = std::filesystem;
using namespace training;

struct TrainOptions {
  std::string lmdb_root = "../data/lmdb/standard_2025_01";
  std::string ckpt_dir = "../checkpoints";
  int epochs = 10;
  int batch_size = 4096;
  int num_workers = 4;
  double learning_rate = 1e-3;
  size_t max_samples = 0; // 0 = use full shard range
};

int main(int argc, char **argv) {
  TrainOptions opt;

  if (argc >= 2)
    opt.lmdb_root = argv[1];
  if (argc >= 3)
    opt.batch_size = std::stoi(argv[2]);
  if (argc >= 4)
    opt.epochs = std::stoi(argv[3]);

  std::cout << "[Train] LMDB root : " << opt.lmdb_root << "\n";
  std::cout << "[Train] Batch size: " << opt.batch_size << "\n";
  std::cout << "[Train] Epochs    : " << opt.epochs << "\n";

  fs::create_directories(opt.ckpt_dir);

  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA, 0);
    std::cout << "[Train] Using CUDA:0\n";
  } else {
    std::cout << "[Train] Using CPU\n";
  }

  // ========================
  // Dataset + Loader
  // ========================
  auto dataset =
      LMDBDataset(opt.lmdb_root).map(torch::data::transforms::Stack<>());

  auto loader = torch::data::make_data_loader(
      std::move(dataset), torch::data::DataLoaderOptions()
                              .batch_size(opt.batch_size)
                              .workers(opt.num_workers)
      // .shuffle(true)  <-- 삭제 (LibTorch C++에 없음)
  );

  // ========================
  // Model
  // ========================
  ChessNet model(12, 64, 6);
  model->to(device);

  std::cout << "[Train] Model parameters: " << model->parameters().size()
            << " tensors\n";

  torch::optim::AdamW optimizer(
      model->parameters(),
      torch::optim::AdamWOptions(opt.learning_rate).weight_decay(1e-4));

  // ========================
  // Training Loop
  // ========================
  for (int epoch = 1; epoch <= opt.epochs; ++epoch) {
    model->train();
    size_t batch_idx = 0;
    double running_loss = 0.0;

    auto ep_start = std::chrono::high_resolution_clock::now();

    for (auto &batch : *loader) {
      auto inputs = batch.data.to(device);
      auto targets = batch.target.to(device).to(torch::kLong);

      optimizer.zero_grad();

      auto [policy_logits, value] = model->forward(inputs);

      auto policy_loss =
          torch::nn::functional::cross_entropy(policy_logits, targets);

      auto loss = policy_loss;
      loss.backward();
      optimizer.step();

      running_loss += loss.item<double>();
      ++batch_idx;

      if (batch_idx % 50 == 0) {
        std::cout << "[Epoch " << epoch << "] Batch " << batch_idx
                  << " | Loss: " << (running_loss / batch_idx) << std::endl;
      }
    }

    auto ep_end = std::chrono::high_resolution_clock::now();
    auto sec =
        std::chrono::duration_cast<std::chrono::milliseconds>(ep_end - ep_start)
            .count() /
        1000.0;

    std::cout << "[Epoch " << epoch << "] Avg Loss: "
              << (running_loss / std::max<size_t>(batch_idx, 1))
              << " | Time: " << sec << "s\n";

    std::string ckpt =
        opt.ckpt_dir + "/chessnet_epoch_" + std::to_string(epoch) + ".pt";

    torch::save(model, ckpt);
    std::cout << "[Train] Saved checkpoint: " << ckpt << "\n";
  }

  std::cout << "[Train] Done.\n";
  return 0;
}
