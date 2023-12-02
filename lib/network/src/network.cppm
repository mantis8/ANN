module;

import Layers;

#include <cstddef>
#include <tuple>

export module Network;

export namespace ann {

template<typename... Layers>
class Network {
  public:
    Network(Layers&&... layers) : layers_{layers...} {};

    template<size_t I = 0>
    decltype(auto) predict(auto X) {
        if constexpr (sizeof...(Layers) > I) {
            // call each layer consecutively
            auto Y = std::get<I>(layers_).predict(X);
            return predict<I + 1>(Y);
        } else {
            return X;
        }
    }

  private:
    std::tuple<Layers...> layers_;
};

} // ann
