import Activations;
import Matrix; 

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("Relu") {
    constexpr size_t size = 3;

    linalg::Matrix<float, size, 1> Z{{-42.42},
                                      {0},
                                      {42.42}};

    SUBCASE("Forward pass") {
        auto A = ann::activations::Relu<float, size>::forward(Z);

        CHECK(doctest::Approx(0)     == A(0, 0));
        CHECK(doctest::Approx(0)     == A(1, 0));
        CHECK(doctest::Approx(42.42) == A(2, 0));
    }

    SUBCASE("Jacobian matrix") {
        auto J = ann::activations::Relu<float, size>::jacobian(Z);
        
        CHECK(doctest::Approx(0) == J(0, 0));
        CHECK(doctest::Approx(0) == J(0, 1));
        CHECK(doctest::Approx(0) == J(0, 2));
        CHECK(doctest::Approx(0) == J(1, 0));
        CHECK(doctest::Approx(0) == J(1, 1));
        CHECK(doctest::Approx(0) == J(1, 2));
        CHECK(doctest::Approx(0) == J(2, 0));
        CHECK(doctest::Approx(0) == J(2, 1));
        CHECK(doctest::Approx(1) == J(2, 2));
    }
}

TEST_CASE("Softmax") {
    constexpr size_t size = 5;
    linalg::Matrix<float, size, 1> Z{{1},
                                     {2},
                                     {3},
                                     {4},
                                     {5}};
    
    SUBCASE("Forward pass") {
        auto A = ann::activations::Softmax<float, size>::forward(Z);

        CHECK(doctest::Approx(0.011656231)  == A(0, 0));
        CHECK(doctest::Approx(0.0316849208) == A(1, 0));
        CHECK(doctest::Approx(0.0861285444) == A(2, 0));
        CHECK(doctest::Approx(0.2341216573) == A(3, 0));
        CHECK(doctest::Approx(0.6364086466) == A(4, 0));
    }
}
