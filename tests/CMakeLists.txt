add_library(${PROJECT_NAME}_Doctest INTERFACE)
target_include_directories(${PROJECT_NAME}_Doctest INTERFACE thirparty)

add_executable(${PROJECT_NAME}_ActivationsTest src/activations_test.cpp)
target_link_libraries(${PROJECT_NAME}_ActivationsTest PRIVATE ${PROJECT_NAME}_Activations
                                                              ${PROJECT_NAME}_Doctest)

add_executable(${PROJECT_NAME}_CostFunctionsTest src/cost_functions_test.cpp)
target_link_libraries(${PROJECT_NAME}_CostFunctionsTest PRIVATE ${PROJECT_NAME}_CostFunctions
                                                                ${PROJECT_NAME}_Doctest)

add_executable(${PROJECT_NAME}_InitializersTest src/initializers_test.cpp)
target_link_libraries(${PROJECT_NAME}_InitializersTest PRIVATE ${PROJECT_NAME}_Initializers
                                                                ${PROJECT_NAME}_Doctest)

add_executable(${PROJECT_NAME}_LayersTest src/layers_test.cpp)
target_link_libraries(${PROJECT_NAME}_LayersTest PRIVATE ${PROJECT_NAME}_Doctest
                                                         ${PROJECT_NAME}_Layers)

add_executable(${PROJECT_NAME}_ModelTest src/model_test.cpp)
target_link_libraries(${PROJECT_NAME}_ModelTest PRIVATE ${PROJECT_NAME}_Doctest
                                                          ${PROJECT_NAME}_Layers
                                                          ${PROJECT_NAME}_Model)

add_test(${PROJECT_NAME}_ActivationsTest ${PROJECT_NAME}_ActivationsTest)
add_test(${PROJECT_NAME}_CostFunctionsTest ${PROJECT_NAME}_CostFunctionsTest)
add_test(${PROJECT_NAME}_InitializersTest ${PROJECT_NAME}_InitializersTest)
add_test(${PROJECT_NAME}_LayersTest ${PROJECT_NAME}_LayersTest)
add_test(${PROJECT_NAME}_ModelTest ${PROJECT_NAME}_ModelTest)
