add_library(${PROJECT_NAME}_Initializers)
target_sources(${PROJECT_NAME}_Initializers
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/concepts.cppm
        src/initializers.cppm
        src/glorot.cppm
)
target_link_libraries(${PROJECT_NAME}_Initializers PUBLIC LinAlg_Matrix)
