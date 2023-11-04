add_library(Activations)
target_sources(Activations
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/activations.cppm
        src/relu.cppm
        src/softmax.cppm
)
