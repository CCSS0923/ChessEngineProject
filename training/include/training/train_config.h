#pragma once
#include <string>

namespace training {

struct TrainConfig {
    std::string lmdbPath{"../data/lmdb/standard_2025_01"};
    int batchSize{256};
    int epochs{1};
    double lr{1e-3};
};

} // namespace training
