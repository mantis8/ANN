module;

#include <algorithm>
#include <type_traits>
	
export module Activations:Relu;

export namespace ann::activations {

struct Relu {
    template<typename T>
    requires std::is_floating_point_v<T>
    static T map(const T& x) {
        return std::max<T>(T{0}, x);
    } 

    template<typename T>
    requires std::is_floating_point_v<T>
    static T derivative(const T& x) {    
        if (T{0} < x) {
            return T{1};
        }
        else {
            return T{0};
        }
    }
};
} // namespace ann::activations
    