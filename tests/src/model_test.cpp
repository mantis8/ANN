import Activations;
import Layers;
import Matrix;
import Model;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Model") {
    auto model = ann::Model{ann::layers::Dense<float, 3, 5, ann::activations::Relu>{},
                            ann::layers::Dense<float, 5, 4, ann::activations::Relu>{},
                            ann::layers::Dense<float, 4, 2, ann::activations::Softmax>{}};

    linalg::Matrix<float, 3, 1> X{};

    auto Y = model.predict(X);

    CHECK(false);
}
