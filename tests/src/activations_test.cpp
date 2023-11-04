import Activations;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

using namespace ann::activations;

TEST_CASE("ReLu") {
    CHECK(0  == Relu<float>::map(0));
    CHECK(42 == Relu<float>::map(42.0));
    CHECK(0  == Relu<float>::map(-42.0));

    CHECK(0 == Relu<float>::derivative(0));
    CHECK(1 == Relu<float>::derivative(42));
    CHECK(0 == Relu<float>::derivative(-42));
}
