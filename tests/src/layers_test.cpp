import Activations;
import Layers;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Dense") {
    ann::layers::Dense<float, 3, 3, ann::activations::Relu> dense{};

    CHECK(false);
}