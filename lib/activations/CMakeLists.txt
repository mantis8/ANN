add_library(${PROJECT_NAME}_Activations)
target_sources(${PROJECT_NAME}_Activations
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/activations.cppm
        src/concepts.cppm
        src/relu.cppm
        src/softmax.cppm
)
target_link_libraries(${PROJECT_NAME}_Activations PUBLIC LinAlg_Matrix)
