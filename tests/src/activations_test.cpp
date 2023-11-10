import Activations;
import Tensor;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace ann::activations;

TEST_CASE("ReLu") {
    CHECK(0  == Relu::map(0.0f));
    CHECK(42 == Relu::map(42.0f));
    CHECK(0  == Relu::map(-42.0f));

    CHECK(0 == Relu::derivative(0.0f));
    CHECK(1 == Relu::derivative(42.0f));
    CHECK(0 == Relu::derivative(-42.0f));
}

TEST_CASE("Softmax") {
    linalg::Tensor<double, 5, 5> T{};
    Softmax::map(T);
}
