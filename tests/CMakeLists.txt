add_library(${PROJECT_NAME}_Doctest INTERFACE)
target_include_directories(${PROJECT_NAME}_Doctest INTERFACE thirparty)

add_executable(${PROJECT_NAME}_ActivationsTest src/activations_test.cpp)
target_link_libraries(${PROJECT_NAME}_ActivationsTest PRIVATE ${PROJECT_NAME}_Activations
                                                              ${PROJECT_NAME}_Doctest)

add_executable(${PROJECT_NAME}_LayersTest src/layers_test.cpp)
target_link_libraries(${PROJECT_NAME}_LayersTest PRIVATE ${PROJECT_NAME}_Layers
                                                         ${PROJECT_NAME}_Doctest)

add_test(${PROJECT_NAME}_Tests ${PROJECT_NAME}_ActivationsTest)
add_test(${PROJECT_NAME}_Tests ${PROJECT_NAME}_LayersTest)
