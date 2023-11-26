import CostFunctions;
import Matrix; 

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"


TEST_CASE("Mse") {
    linalg::Matrix<float, 3, 1> Y{{1},
                                  {2},
                                  {3}};

    linalg::Matrix<float, 3, 1> Y_hat{{1.1},
                                      {1.9},
                                      {3.5}};

    SUBCASE("Calculate cost") {
        auto c = ann::cost_functions::Mse::map(Y, Y_hat);
        CHECK(doctest::Approx(0.09) == c);
    }

    SUBCASE("Jacobian matrix") {
        auto J = ann::cost_functions::Mse::jacobian(Y, Y_hat);
        CHECK(doctest::Approx(0.2f / 3.0f)  == J(0, 0));
        CHECK(doctest::Approx(-0.2f / 3.0f) == J(0, 1));
        CHECK(doctest::Approx(1.0f / 3.0f)  == J(0, 2));
    }
}
