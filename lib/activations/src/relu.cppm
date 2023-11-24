module;

#include <algorithm>
#include <type_traits>

export module Activations:Relu;

import Matrix;

export namespace ann::activations {

class Relu {
  public:
    template<typename T, size_t Size>
    requires std::is_floating_point_v<T>
    static constexpr linalg::Matrix<T, Size, 1> map(const linalg::Matrix<T, Size, 1>& Z) {
        linalg::Matrix<T, Size, 1> A{};
        std::transform(Z.cbegin(), Z.cend(), A.begin(), relu<T>);

        return A;
    };
    
    template<typename T, size_t Size>
    requires std::is_floating_point_v<T>
    static constexpr linalg::Matrix<T, Size, Size> jacobian(const linalg::Matrix<T, Size, 1>& Z) {
        // TODO make use of diagonal matrix
        linalg::Matrix<T, Size, Size> J{};
        
        for (size_t i = 0; i < Size; i++) {
            J(i, i) = reluDerivative(Z(i, 0));
        }

        return J;
    };
    
  private:
    template<typename T>
    static constexpr T relu(const T z) {
        return std::max<T>(T{0}, z);
    }; 

    template<typename T>
    static constexpr T reluDerivative(const T z) {    
        if (T{0} < z) {
            return T{1};
        }
        else {
            return T{0};
        }
    }; 
};
} // namespace ann::activations
    