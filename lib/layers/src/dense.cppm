module;

#include <cstddef>

import Matrix;
import Tensor;

import Activations;

export module Layers:Dense;

export namespace ann::layers {

template<typename T, size_t Inputs, size_t Outputs, typename Activation>
requires activations::is_activation<T, Outputs, Activation>
class Dense {
  public:
    Dense() = default;
    Dense(const linalg::Matrix<T, Outputs, Inputs>& W, const linalg::Matrix<T, Outputs, 1>& B) : W_{W}, B_{B} {};

    linalg::Matrix<T, Outputs, 1> feed(const linalg::Matrix<T, Inputs, 1>& X) {
        return Activation::map(W_ * X + B_);
    };

    linalg::Tensor<T, Outputs, 1> train(const linalg::Tensor<T, Inputs, 1>& X) {
        // TODO check design
        //auto Z = W_ * X + B_;
        //auto A = Activation::map(Z);
        //auto J = Activation::jacobian(Z);
        linalg::Tensor<T, Outputs, 1> Y{};

        return Y;
    };

    // TODO maybe use ducktyping instead of inheritance
    void update(const T lr, const linalg::Matrix<T, Outputs, Inputs>& dW, const linalg::Matrix<T, Outputs, 1>& dB) {
        W_ = W_ - lr * dW;
        B_ = B_ - lr * dB;
    }

  private:
    linalg::Matrix<T, Outputs, Inputs> W_;
    linalg::Matrix<T, Outputs, 1> B_;
};

} // ann::layers
