module;

#include <cstddef>

export module Layers:Dense;

import Matrix;
import Tensor;

import :Interface;

template<typename T, size_t Size, typename Activation>
concept is_activation = requires(Activation a) {
    {a.template forward(linalg::Matrix<T, Size, 1>{})};
    {a.template jacobian(linalg::Matrix<T, Size, 1>{})};
};


export namespace ann::layers {

template<typename T, size_t Inputs, size_t Outputs, typename Activation>
requires is_activation<T, Outputs, Activation>
class Dense: ILayer<T, Inputs, Outputs> {
    linalg::Matrix<T, Outputs, 1> predict(linalg::Matrix<T, Inputs, 1> input) override {
        // TODO implement
        auto A = Activation::forward(input);
        linalg::Matrix<T, Outputs, 1> Y{};
        return Y;
    };

    linalg::Tensor<T, Outputs, 1> train(linalg::Tensor<T, Inputs, 1>) override {
        linalg::Tensor<T, Outputs, 1> Y{};
        // TODO implement
        return Y;
    };

  private:
    linalg::Matrix<T, Outputs, Inputs> weights_;
    linalg::Matrix<T, Outputs, 1> bias_;
};

} // ann::layers
