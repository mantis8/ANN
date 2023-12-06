module;

import Matrix;

#include <cmath>
#include <numeric>
#include <type_traits>

export module CostFunctions:Mse;

export namespace ann::cost_functions {

class Mse {
  public:
    template<typename T, size_t Inputs>
    requires std::is_floating_point_v<T> && (Inputs > 0)
    static T map(const linalg::Matrix<T, Inputs, 1> Y, const linalg::Matrix<T, Inputs, 1>& Y_hat) {
        linalg::Matrix<T, Inputs, 1> L{};

        // TODO use zip_transform
        for (size_t i = 0; i < Inputs; i++) {
            L(i, 0) = mse(Y(i, 0), Y_hat(i, 0));
        }

        return std::reduce(L.cbegin(), L.cend()) / Inputs;
    };
    
    template<typename T, size_t Inputs>
    requires std::is_floating_point_v<T> && (Inputs > 0)
    static linalg::Matrix<T, 1, Inputs> jacobian(const linalg::Matrix<T, Inputs, 1> Y, const linalg::Matrix<T, Inputs, 1>& Y_hat) {
        // TODO use diagonal matrix
        linalg::Matrix<T, 1, Inputs> J{};
        
        // TODO use zip_transform
        for (size_t i = 0; i < Inputs; i++) {
            J(0, i) = mseDerivative(Y(0, i), Y_hat(0, i)) / Inputs;
        }

        return J;
    };
    
  private:
    template<typename T>
    static T mse(const T y, const T y_hat) {
        return std::pow((y - y_hat), 2u);
    }; 

    template<typename T>
    static T mseDerivative(const T y, const T y_hat) {    
        return -2.0f * (y - y_hat);
    };
};
} // namespace ann::cost_functions