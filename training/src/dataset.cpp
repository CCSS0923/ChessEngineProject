#include "training/dataset.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static std::string join(const std::string& a, const std::string& b) {
    return a + "/" + b;
}

LMDBDataset::LMDBDataset(const std::string& root) {
    load_shards(root);

    total_samples_ = 0;
    for (auto& s : shards_) {
        total_samples_ += s.num_samples;
    }

    if (shards_.empty()) {
        std::cerr << "[WARN] No shards found under " << root << "\n";
    }
}

void LMDBDataset::load_shards(const std::string& root) {
    shards_.clear();

    for (auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) continue;

        std::string name = entry.path().filename().string();
        if (name.rfind("shard_", 0) != 0) continue;

        std::string meta_path = join(entry.path().string(), "meta.json");
        if (!std::filesystem::exists(meta_path)) {
            std::cerr << "[WARN] meta.json missing in " << name << "\n";
            continue;
        }

        std::ifstream f(meta_path);
        json meta;
        f >> meta;

        ShardInfo info;
        info.shard_index = meta.value("shard_index", 0);
        info.shard_path = entry.path().string();
        info.num_samples = meta["num_samples"].get<int64_t>();
        info.global_start = meta["sample_range"]["global_start"].get<int64_t>();
        info.global_end   = meta["sample_range"]["global_end"].get<int64_t>();

        shards_.push_back(info);
    }

    std::sort(shards_.begin(), shards_.end(),
        [](const ShardInfo& a, const ShardInfo& b) {
            return a.shard_index < b.shard_index;
        }
    );
}

bool LMDBDataset::find_shard_for_index(int64_t global_index, int& shard_idx, int64_t& local_index) {
    for (size_t i = 0; i < shards_.size(); i++) {
        if (global_index >= shards_[i].global_start &&
            global_index <= shards_[i].global_end)
        {
            shard_idx = static_cast<int>(i);
            local_index = global_index - shards_[i].global_start;
            return true;
        }
    }
    return false;
}

torch::data::Example<> LMDBDataset::get(size_t index) {
    int shard_idx;
    int64_t local_index;

    if (!find_shard_for_index(static_cast<int64_t>(index), shard_idx, local_index)) {
        throw std::runtime_error("Index not found in any shard.");
    }

    // Placeholder: return zero tensor. To use real data, add msgpack decoding of LMDB values here.
    auto x = torch::zeros({12, 8, 8}, torch::kFloat32);
    auto y = torch::tensor(0, torch::kLong);
    return {x, y};
}
