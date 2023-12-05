module;

import Matrix;

#include <concepts>
#include <cstddef>

export module Initializers:Concepts;

export namespace ann::initializers {

template<typename T, size_t Inputs, size_t Outputs, typename Initializer>
concept is_initializer = requires(Initializer i, linalg::Matrix<T, Outputs, Inputs> W) {
    {i.template initialize(W)} -> std::same_as<void>;
};

} // ann::initializer
