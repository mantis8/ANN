module;

import Matrix;

#include <cstddef>

export module Layers:Concepts;

export namespace ann::layers {

template<typename T, size_t Size, typename Layer>
concept is_layer = requires(Layer l) {
    {l.template feed(linalg::Matrix<T, Size, 1>{})};
};

} // ann::layers
