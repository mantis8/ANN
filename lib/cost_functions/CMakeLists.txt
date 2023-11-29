add_library(${PROJECT_NAME}_CostFunctions)
target_sources(${PROJECT_NAME}_CostFunctions
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/concepts.cppm
        src/cost_functions.cppm
        src/mse.cppm
)
target_link_libraries(${PROJECT_NAME}_CostFunctions PUBLIC LinAlg_Matrix)
