module;

#include <cstddef>

export module CostFunctions:Concepts;

import Matrix;

export namespace ann::cost_functions {

template<typename T, size_t Size, typename CostFunction>
concept is_cost_function = requires(CostFunction c) {
    {c.template map(linalg::Matrix<T, Size, 1>{}, linalg::Matrix<T, Size, 1>{})};
    {c.template jacobian(linalg::Matrix<T, Size, 1>{}, linalg::Matrix<T, Size, 1>{})};
};

} // ann::cost_functions
