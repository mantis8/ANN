add_library(${PROJECT_NAME}_Layers)
target_sources(${PROJECT_NAME}_Layers
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/concepts.cppm
        src/dense.cppm
        src/layers.cppm
)
target_link_libraries(${PROJECT_NAME}_Layers PUBLIC ${PROJECT_NAME}_Activations
                                                    LinAlg_Tensor)
