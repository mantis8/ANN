import Activations;
import Layers;
import Matrix;
import Network;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Network") {
    auto n = ann::Network{ann::layers::Dense<float, 3, 5, ann::activations::Relu>{},
                          ann::layers::Dense<float, 5, 4, ann::activations::Relu>{},
                          ann::layers::Dense<float, 4, 2, ann::activations::Softmax>{}};

    linalg::Matrix<float, 3, 1> X{};

    auto Y = n.predict(X);

    CHECK(false);
}
