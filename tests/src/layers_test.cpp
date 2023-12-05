import Activations;
import Initializers;
import Layers;
import Matrix;

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Dense") {
    SUBCASE("feed") {
        linalg::Matrix<float, 3, 3> W{{ 1.2,  3.1, -0.3},
                                      {-4.4, -0.4,  3.7},
                                      { 2.5, -1.9, -2.1}};

        linalg::Matrix<float, 3, 1> B{{-21},
                                      { 20},
                                      { 2}};

        ann::layers::Dense<float, 3, 3, ann::activations::Softmax, ann::initializers::Glorot> dense{W, B};
        
        linalg::Matrix<float, 3, 1> X{{0.5},
                                      {7.2},
                                      {-4.3}};

        auto Y = dense.feed(X);

        CHECK(doctest::Approx(0.9756598287) == Y(0, 0)); 
        CHECK(doctest::Approx(0.0146305819) == Y(1, 0)); 
        CHECK(doctest::Approx(0.0097095893) == Y(2, 0)); 
    }
}