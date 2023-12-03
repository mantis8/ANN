module;

import Matrix;

#include <cstddef>

export module Initializers:Concepts;

export namespace ann::initializers {

template<typename T, size_t Size, typename Initializer>
concept is_initializer = requires(Initializer i) {
    // TODO implement
    {i.template map(linalg::Matrix<T, Size, 1>{})};
    {i.template jacobian(linalg::Matrix<T, Size, 1>{})};
};

} // ann::initializer
