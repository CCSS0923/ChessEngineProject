#include "training/model.h"

namespace training {

ResidualBlockImpl::ResidualBlockImpl(int channels) {
    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3)
                .padding(1)
                .bias(false)
        )
    );
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(channels));

    conv2 = register_module(
        "conv2",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, channels, 3)
                .padding(1)
                .bias(false)
        )
    );
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(channels));
}

torch::Tensor ResidualBlockImpl::forward(const torch::Tensor& x) {
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);

    out = conv2->forward(out);
    out = bn2->forward(out);

    out = torch::relu(out + x); // skip connection
    return out;
}

ChessNetImpl::ChessNetImpl(int in_channels, int channels, int num_blocks) {
    // Stem: 3×3 conv + BN + ReLU
    stem_conv = register_module(
        "stem_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, channels, 3)
                .padding(1)
                .bias(false)
        )
    );
    stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(channels));

    // Residual blocks
    for (int i = 0; i < num_blocks; ++i) {
        residual_blocks->push_back(ResidualBlock(channels));
    }
    register_module("residual_blocks", residual_blocks);

    // Policy head: 1×1 conv → ReLU → flatten → Linear → logits
    policy_conv = register_module(
        "policy_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, 32, 1)
                .bias(true)
        )
    );
    // 32 × 8 × 8 = 2048
    policy_fc = register_module(
        "policy_fc",
        torch::nn::Linear(32 * 8 * 8, POLICY_SIZE)
    );

    // Value head: 1×1 conv → ReLU → flatten → FC → ReLU → FC → tanh
    value_conv = register_module(
        "value_conv",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(channels, 32, 1)
                .bias(true)
        )
    );
    value_fc1 = register_module(
        "value_fc1",
        torch::nn::Linear(32 * 8 * 8, 128)
    );
    value_fc2 = register_module(
        "value_fc2",
        torch::nn::Linear(128, 1)
    );
}

std::pair<torch::Tensor, torch::Tensor> ChessNetImpl::forward(const torch::Tensor& x) {
    // x: [N, 12, 8, 8]
    auto out = stem_conv->forward(x);
    out = stem_bn->forward(out);
    out = torch::relu(out);

    out = residual_blocks->forward(out);

    // Policy head
    auto p = policy_conv->forward(out);
    p = torch::relu(p);
    p = p.view({p.size(0), -1}); // [N, 2048]
    auto policy_logits = policy_fc->forward(p); // [N, POLICY_SIZE]

    // Value head
    auto v = value_conv->forward(out);
    v = torch::relu(v);
    v = v.view({v.size(0), -1}); // [N, 2048]
    v = value_fc1->forward(v);
    v = torch::relu(v);
    v = value_fc2->forward(v);
    auto value = torch::tanh(v); // [-1, 1]

    return {policy_logits, value};
}

} // namespace training
