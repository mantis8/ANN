module;

import Matrix;

#include <cmath>
#include <numeric>
#include <type_traits>

export module CostFunctions:Mse;

export namespace ann::cost_functions {

class Mse {
  public:
    template<typename T, size_t Size>
    requires std::is_floating_point_v<T> && (Size > 0)
    static T map(const linalg::Matrix<T, Size, 1> Y, const linalg::Matrix<T, Size, 1>& Y_hat) {
        linalg::Matrix<T, Size, 1> L{};

        // TODO use zip_transform
        for (size_t i = 0; i < Size; i++) {
            L(i, 0) = mse(Y(i, 0), Y_hat(i, 0));
        }

        return std::reduce(L.cbegin(), L.cend()) / Size;
    };
    
    template<typename T, size_t Size>
    requires std::is_floating_point_v<T>
    static linalg::Matrix<T, 1, Size> jacobian(const linalg::Matrix<T, Size, 1> Y, const linalg::Matrix<T, Size, 1>& Y_hat) {
        // TODO use diagonal matrix
        linalg::Matrix<T, 1, Size> J{};
        
        // TODO use zip_transform
        for (size_t i = 0; i < Size; i++) {
            J(0, i) = mseDerivative(Y(0, i), Y_hat(0, i)) / Size;
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