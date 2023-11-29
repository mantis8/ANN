add_library(${PROJECT_NAME}_Network)
target_sources(${PROJECT_NAME}_Network
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/network.cppm
)
target_link_libraries(${PROJECT_NAME}_Network PUBLIC ${PROJECT_NAME}_Layers)
