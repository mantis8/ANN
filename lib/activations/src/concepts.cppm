module;

import Matrix;

#include <concepts>
#include <cstddef>

export module Activations:Concepts;

export namespace ann::activations {

template<typename T, size_t Inputs, typename Activation>
concept is_activation = requires(Activation a, linalg::Matrix<T, Inputs, 1> Z) {
    {a.template map(Z)} -> std::same_as<linalg::Matrix<T, Inputs, 1>>;
    {a.template jacobian(Z)} -> std::same_as<linalg::Matrix<T, Inputs, Inputs>>;
};

} // ann::activations
