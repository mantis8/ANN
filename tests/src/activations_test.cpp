import Activations;
import Matrix; 

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <numeric>

TEST_CASE("Relu") {
    constexpr size_t size = 3;

    linalg::Matrix<float, size, 1> Z{{-42.42},
                                      {0},
                                      {42.42}};

    SUBCASE("Forward pass") {
        auto A = ann::activations::Relu::map(Z);

        CHECK(doctest::Approx(0)     == A(0, 0));
        CHECK(doctest::Approx(0)     == A(1, 0));
        CHECK(doctest::Approx(42.42) == A(2, 0));
    }

    SUBCASE("Jacobian matrix") {
        auto J = ann::activations::Relu::jacobian(Z);
        
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
    constexpr size_t size = 3;
    linalg::Matrix<double, size, 1> Z{{1},
                                      {2},
                                      {3}};
    
    SUBCASE("Forward pass") {
        auto A = ann::activations::Softmax::map(Z);

        CHECK(doctest::Approx(0.0900305732) == A(0, 0));
        CHECK(doctest::Approx(0.2447284711) == A(1, 0));
        CHECK(doctest::Approx(0.6652409558) == A(2, 0));

        double sum = std::reduce(A.cbegin(), A.cend());    
        CHECK(doctest::Approx(1) == sum);
    }

    SUBCASE("Jacobain matrix") {
        auto J = ann::activations::Softmax::jacobian(Z);

        // 1st row
        CHECK(doctest::Approx(0.08192506908927944176)  == J(0, 0));
        CHECK(doctest::Approx(-0.02203304453149263452) == J(0, 1));
        CHECK(doctest::Approx(-0.05989202456678986456) == J(0, 2));

            
        // 2nd row
        CHECK(doctest::Approx(-0.02203304453149263452) == J(1, 0));
        CHECK(doctest::Approx(0.18483644653305646479)  == J(1, 1));
        CHECK(doctest::Approx(-0.16280340202603667738) == J(1, 2));

        // 3rd row
        CHECK(doctest::Approx(-0.05989202456678986456) == J(2, 0));
        CHECK(doctest::Approx(-0.16280340202603667738) == J(2, 1));
        CHECK(doctest::Approx(0.22269542652630244636)  == J(2, 2));
    }
}
