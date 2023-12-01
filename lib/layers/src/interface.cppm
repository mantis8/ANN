module;

#include <cstddef>

import Matrix;
import Tensor;

export module Layers:Interface;

export namespace ann::layers {
template<typename T, size_t Inputs, size_t Outputs>
struct ILayer {
    virtual linalg::Matrix<T, Outputs, 1> predict(const linalg::Matrix<T, Inputs, 1>&) = 0;
    virtual linalg::Tensor<T, Outputs, 1> train(const linalg::Tensor<T, Inputs, 1>&) = 0;
};

} // ann::layers
