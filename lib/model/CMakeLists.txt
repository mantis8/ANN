add_library(${PROJECT_NAME}_Model)
target_sources(${PROJECT_NAME}_Model
    PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        src/model.cppm
)
target_link_libraries(${PROJECT_NAME}_Model PUBLIC ${PROJECT_NAME}_Layers)
