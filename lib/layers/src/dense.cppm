module;

#include <cstddef>

export module Layers:Dense;



import Matrix;
import Tensor;

import Activations;
import :Interface;

export namespace ann::layers {

template<typename T, size_t Inputs, size_t Outputs, typename Activation>
requires activations::is_activation<T, Outputs, Activation>
class Dense: public ILayer<T, Inputs, Outputs> {
    linalg::Matrix<T, Outputs, 1> predict(const linalg::Matrix<T, Inputs, 1>& X) override {
        return Activation::map(W_ * X + B_);
    };

    linalg::Tensor<T, Outputs, 1> train(const linalg::Tensor<T, Inputs, 1>& X) override {
        // TODO check design
        //auto Z = W_ * X + B_;
        //auto A = Activation::map(Z);
        //auto J = Activation::jacobian(Z);

        return X;
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
