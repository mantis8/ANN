module;

import Matrix;

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <valarray>

export module Activations:Softmax;

export namespace ann::activations {

template<typename T, size_t Size>
requires std::is_floating_point_v<T>

class Softmax {
  public:
    static linalg::Matrix<T, Size, 1> forward(linalg::Matrix<T, Size, 1> Z) {
        linalg::Matrix<T, Size, 1> A{};
        std::transform(Z.cbegin(), Z.cend(), A.begin(), exp);

        //T sum = T{0};
        //for (const auto& elem : A) {
        //    sum += elem;
        //}

        //auto divide = 0

        //std::transform(A.begin(), A.end(), A.begin(), )


        return A;
    };
    
    static linalg::Matrix<T, Size, Size> jacobian(linalg::Matrix<T, Size, 1> Z) {
        // TODO make use of diagonal matrix
        linalg::Matrix<T, Size, Size> jacobian{};
        

        return jacobian;
    };
    
  private:
    static T exp(const T& z) {
        return std::exp(z);
    };

    static T derive(const T& z) {    

    }; 
};
} // namespace ann::activations
    