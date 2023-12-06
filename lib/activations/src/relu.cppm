module;

import Matrix;

#include <algorithm>
#include <type_traits>

export module Activations:Relu;

export namespace ann::activations {

class Relu {
  public:
    template<typename T, size_t Inputs>
    requires std::is_floating_point_v<T>
    static linalg::Matrix<T, Inputs, 1> map(const linalg::Matrix<T, Inputs, 1>& Z) {
        linalg::Matrix<T, Inputs, 1> A{};
        std::transform(Z.cbegin(), Z.cend(), A.begin(), relu<T>);

        return A;
    };
    
    template<typename T, size_t Inputs>
    requires std::is_floating_point_v<T>
    static linalg::Matrix<T, Inputs, Inputs> jacobian(const linalg::Matrix<T, Inputs, 1>& Z) {
        // TODO make use of diagonal matrix
        linalg::Matrix<T, Inputs, Inputs> J{};
        
        for (size_t i = 0; i < Inputs; i++) {
            J(i, i) = reluDerivative(Z(i, 0));
        }

        return J;
    };
    
  private:
    template<typename T>
    static T relu(const T z) {
        return std::max<T>(T{0}, z);
    }; 

    template<typename T>
    static T reluDerivative(const T z) {    
        if (T{0} < z) {
            return T{1};
        }
        else {
            return T{0};
        }
    }; 
};
} // namespace ann::activations
    