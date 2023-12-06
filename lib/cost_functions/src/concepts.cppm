module;

import Matrix;

#include <concepts>
#include <cstddef>

export module CostFunctions:Concepts;

export namespace ann::cost_functions {

template<typename T, size_t Inputs, typename CostFunction>
concept is_cost_function = requires(CostFunction c, linalg::Matrix<T, Inputs, 1> Y, linalg::Matrix<T, Inputs, 1> Y_hat) {
    {c.template map(Y, Y_hat)} -> std::same_as<T>;
    {c.template jacobian(Y, Y_hat)} -> std::same_as<linalg::Matrix<T, 1, Inputs>>;
};

} // ann::cost_functions
