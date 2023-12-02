add_library(${PROJECT_NAME}_Doctest INTERFACE)
target_include_directories(${PROJECT_NAME}_Doctest INTERFACE thirparty)

add_executable(${PROJECT_NAME}_ActivationsTest src/activations_test.cpp)
target_link_libraries(${PROJECT_NAME}_ActivationsTest PRIVATE ${PROJECT_NAME}_Activations
                                                              ${PROJECT_NAME}_Doctest)

add_executable(${PROJECT_NAME}_CostFunctionsTest src/cost_functions_test.cpp)
target_link_libraries(${PROJECT_NAME}_CostFunctionsTest PRIVATE ${PROJECT_NAME}_CostFunctions
                                                                ${PROJECT_NAME}_Doctest)

add_executable(${PROJECT_NAME}_LayersTest src/layers_test.cpp)
target_link_libraries(${PROJECT_NAME}_LayersTest PRIVATE ${PROJECT_NAME}_Doctest
                                                         ${PROJECT_NAME}_Layers)

add_executable(${PROJECT_NAME}_NetworkTest src/network_test.cpp)
target_link_libraries(${PROJECT_NAME}_NetworkTest PRIVATE ${PROJECT_NAME}_Doctest
                                                          ${PROJECT_NAME}_Layers
                                                          ${PROJECT_NAME}_Network)

add_test(ActivationsTest ${PROJECT_NAME}_ActivationsTest)
add_test(CostFunctionsTest ${PROJECT_NAME}_CostFunctionsTest)
add_test(LayersTest ${PROJECT_NAME}_LayersTest)
add_test(NetworkTest ${PROJECT_NAME}_NetworkTest)
