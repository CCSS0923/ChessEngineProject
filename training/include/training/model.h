#pragma once

#include <torch/torch.h>

namespace training {

// 정책 차원 (from 64 × to 64 + 프로모션 등 압축 인덱스)
constexpr int POLICY_SIZE = 4672;

// 3×3 Residual Block
struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};

    ResidualBlockImpl(int channels);

    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(ResidualBlock);

// 메인 정책+밸류 네트워크
struct ChessNetImpl : torch::nn::Module {
    // Stem
    torch::nn::Conv2d stem_conv{nullptr};
    torch::nn::BatchNorm2d stem_bn{nullptr};

    // Residual blocks
    torch::nn::Sequential residual_blocks;

    // Policy head
    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::Linear policy_fc{nullptr};

    // Value head
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};

    ChessNetImpl(int in_channels = 12, int channels = 64, int num_blocks = 6);

    // 반환: {policy_logits, value_scalar}
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x);
};
TORCH_MODULE(ChessNet);

} // namespace training
