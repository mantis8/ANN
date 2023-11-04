import Activations;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("simple test") {
    ann::activations::Relu relu{};
    ann::activations::Softmax softmax{};

    CHECK(true);   
}
