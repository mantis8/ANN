module;

import Matrix;

#include <algorithm>
#include <cmath>
#include <random>

export module Initializers:Glorot;

export namespace ann::initializers {

struct Glorot {
    template<typename T, size_t Inputs, size_t Outputs>
    requires std::is_floating_point_v<T>
    static void initialize(linalg::Matrix<T, Outputs, Inputs>& W) {
        std::random_device rd{};
        std::mt19937 gen{rd()};

        constexpr auto r = std::sqrt(6 / (Inputs + Outputs));
        std::uniform_real_distribution<> dis(-r, r);

        std::for_each(W.begin(), W.end(), dis(gen));
    }
};

} // namespace ann::initializers
