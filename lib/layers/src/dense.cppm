module;

#include <cstddef>

export module Layers:Dense;

import Matrix;
import Tensor;

import :Interface;

export namespace ann::layers {

template<typename T, size_t Inputs, size_t Outputs, typename Activation>
// TODO add concepts
class Dense: ILayer<T, Inputs, Outputs> {
    linalg::Matrix<T, Outputs, 1> predict(linalg::Matrix<T, Inputs, 1>) override {
        // TODO implements
        linalg::Matrix<T, Outputs, 1> Y{};
        return Y;
    };

    linalg::Tensor<T, Outputs, 1> train(linalg::Tensor<T, Inputs, 1>) override {
        linalg::Matrix<T, Outputs, 1> Y{};
        return Y;
    };

  private:

};

} // ann::layers
