module;

#include <algorithm>
#include <type_traits>
	
export module Activations:Relu;

export namespace ann::activations {

template<typename T>
class Relu { 
  public:
    static T map(const T& x) {
        return std::max<T>(T{0}, x);
    } 

    static T derivative(const T& x) {    
        if (T{0} < x) {
            return T{1};
        }
        else {
            return T{0};
        }
    } 
  private:
    static_assert(std::is_scalar<T>(), "Only scalar types are allowed.");
};
} // namespace ann::activations
    