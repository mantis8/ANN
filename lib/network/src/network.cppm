module;

#include <cstddef>
#include <tuple>

export module Network;

import Layers;

export namespace ann::network {

template<size_t T, typename... Layers>
class Network {
  public:
    Network();
 
    template<size_t I = 0>
    constexpr void predict() {
        if constexpr(I ==std::tuple_size<std::tuple<T...>>{}) {
            return;
        }

        else {
            std::get<I>(layers_);
            predict<I+1>();
        } 
    }

  private:
    std::tuple<Layers...> layers_;
}

} // ann::network
