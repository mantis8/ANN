module;

#include <cstddef>

export module Activations:Concepts;

import Matrix;

export namespace ann::activations {

template<typename T, size_t Size, typename Activation>
concept is_activation = requires(Activation a) {
    {a.template map(linalg::Matrix<T, Size, 1>{})};
    {a.template jacobian(linalg::Matrix<T, Size, 1>{})};
};

} // ann::activations
