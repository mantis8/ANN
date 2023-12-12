import Activations;
import Initializers;
import Layers;
import Matrix;
import Model;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Model") {
    linalg::Matrix<float, 3, 2> W1{{-2.2,  0.1},
                                   { 1,    0.7},
                                   { 1.5, -1.3}};

    linalg::Matrix<float, 3, 1> B1{{-1},
                                   { 1},
                                   {-1}};

    linalg::Matrix<float, 2, 3> W2{{-4.2, 0.4,  1},
                                   { 2.2, 0,   -1}};

    linalg::Matrix<float, 2, 1> B2{{-0.1},
                                   { 0.1}};

    auto model = ann::Model{ann::layers::Dense<float, 2, 3, ann::activations::Relu>{W1, B1},
                            ann::layers::Dense<float, 3, 2, ann::activations::Softmax>{W2, B2}};

    linalg::Matrix<float, 2, 1> X{{1.1},
                                  {2.2}};

    auto Y = model.predict(X);

    CHECK(doctest::Approx(0.7783367616) == Y(0, 0));
    CHECK(doctest::Approx(0.2216632384) == Y(1, 0));
}
