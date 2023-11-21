module;

#include <algorithm>
#include <type_traits>

export module Activations:Relu;

import Matrix;

export namespace ann::activations {

template<typename T, size_t Size>
requires std::is_floating_point_v<T>

class Relu {
  public:
    static linalg::Matrix<T, Size, 1> forward(const linalg::Matrix<T, Size, 1>& Z) {
        linalg::Matrix<T, Size, 1> A{};
        std::transform(Z.cbegin(), Z.cend(), A.begin(), map);

        return A;
    };
    
    static linalg::Matrix<T, Size, Size> jacobian(const linalg::Matrix<T, Size, 1>& Z) {
        // TODO make use of diagonal matrix
        linalg::Matrix<T, Size, Size> J{};
        
        for (size_t i = 0; i < Size; i++) {
            J(i, i) = derivative(Z(i, 0));
        }

        return J;
    };
    
  private:
    static T map(const T z) {
        return std::max<T>(T{0}, z);
    }; 

    static T derivative(const T z) {    
        if (T{0} < z) {
            return T{1};
        }
        else {
            return T{0};
        }
    }; 
};
} // namespace ann::activations
    