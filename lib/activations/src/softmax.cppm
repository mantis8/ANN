module;

import Tensor;

#include <type_traits>
#include <cstddef>

export module Activations:Softmax;

export namespace ann::activations {

struct Softmax {
    template<typename T, size_t Rows, size_t Columns>
    requires std::is_floating_point_v<T>
    static T map(const linalg::Tensor<T, Rows, Columns>& x) {
        return T{};
    } 

    template<typename T, size_t Rows, size_t Columns>
    requires std::is_floating_point_v<T>
    static T derivative(const T& x) { 
        return T{};      
    } 
};
} // namespace ann::activations
    