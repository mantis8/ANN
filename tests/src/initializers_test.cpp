import Initializers;
import Matrix;

#include <iostream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

bool inRange(auto x, auto a, auto b) {
    return (a <= x) && (x <= b);
}

TEST_CASE("Glorot") {
    constexpr size_t n = 2;
    constexpr size_t m = 2;

    linalg::Matrix<float, n, m> W{{42, 42},
                                  {42, 42}};

    float r = std::sqrt(6.0f / (n + m));

    CHECK(!inRange(42, -r, r));

    ann::initializers::Glorot::initialize(W);

    CHECK(inRange(W(0, 0), -r, r));
    CHECK(inRange(W(0, 1), -r, r));
    CHECK(inRange(W(1, 0), -r, r));
    CHECK(inRange(W(1, 1), -r, r));
}