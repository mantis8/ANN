module;

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

export module Activations:Softmax;

import Matrix;

export namespace ann::activations {

template<typename T, size_t Size>
requires std::is_floating_point_v<T>

class Softmax {
  public:
    static linalg::Matrix<T, Size, 1> forward(const linalg::Matrix<T, Size, 1>& Z) {
        linalg::Matrix<T, Size, 1> A{};
        std::transform(Z.cbegin(), Z.cend(), A.begin(), exp);

        const T sum = std::reduce(A.cbegin(), A.cend());
        std::for_each(A.begin(), A.end(), [sum](T& z){
            z = z / sum;
        });

        return A;
    };
    
    static linalg::Matrix<T, Size, Size> jacobian(const linalg::Matrix<T, Size, 1>& Z) {
        auto A = forward(Z);
        
        linalg::Matrix<T, Size, Size> J{};
        for (size_t i = 0; i < Size; i++) {
            for (size_t j = i; j < Size; j++) {
                if (i == j) {
                    J(i, j) = A(i, 0) * (1 - A(i, 0));
                } else {
                    J(i, j) = -1 * (A(i, 0) * A(j, 0));
                    J(j, i) = J(i, j);
                }
            }
        }

        return J;
    };
    
  private:
    static T exp(const T& z) {
        return std::exp(z);
    };
};
} // namespace ann::activations
    