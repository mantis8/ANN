module;

import Matrix;

#include <concepts>
#include <cstddef>

export module Layers:Concepts;

export namespace ann::layers {

template<typename T, size_t Inputs, size_t Outputs, typename Layer>
concept is_layer = requires(Layer l, linalg::Matrix<T, Inputs, 1> X) {
    {l.template feed(X)} -> std::same_as<linalg::Matrix<T, Outputs, 1> >;
};

} // ann::layers
