module;

import Matrix;

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>

export module Activations:Softmax;

export namespace ann::activations {

class Softmax {
  public:
    template<typename T, size_t Inputs>
    requires std::is_floating_point_v<T>
    static linalg::Matrix<T, Inputs, 1> map(const linalg::Matrix<T, Inputs, 1>& Z) {
        linalg::Matrix<T, Inputs, 1> A{};
        std::transform(Z.cbegin(), Z.cend(), A.begin(), exp<T>);

        const T sum = std::reduce(A.cbegin(), A.cend());
        std::for_each(A.begin(), A.end(), [sum](T& z){
            z = z / sum;
        });

        return A;
    };
    
    template<typename T, size_t Inputs>
    requires std::is_floating_point_v<T>
    static linalg::Matrix<T, Inputs, Inputs> jacobian(const linalg::Matrix<T, Inputs, 1>& Z) {
        // TODO maybe refactor to avoid second call to map
        auto A = map(Z);
        
        linalg::Matrix<T, Inputs, Inputs> J{};
        for (size_t i = 0; i < Inputs; i++) {
            for (size_t j = i; j < Inputs; j++) {
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
    template<typename T>
    static T exp(const T& z) {
        return std::exp(z);
    };
};
} // namespace ann::activations
    