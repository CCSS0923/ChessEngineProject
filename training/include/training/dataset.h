#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <vector>

struct ShardInfo {
    int shard_index;
    std::string shard_path;
    int64_t num_samples;
    int64_t global_start;
    int64_t global_end;
};

// LMDB-backed dataset; currently returns zeroed tensors placeholder until msgpack decoding is added.
class LMDBDataset : public torch::data::Dataset<LMDBDataset> {
public:
    explicit LMDBDataset(const std::string& root);

    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override {
        return total_samples_;
    }

private:
    std::vector<ShardInfo> shards_;
    int64_t total_samples_{0};

    void load_shards(const std::string& root);
    bool find_shard_for_index(int64_t global_index, int& shard_idx, int64_t& local_index);
};
