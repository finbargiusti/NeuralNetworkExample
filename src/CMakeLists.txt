cmake_minimum_required(VERSION 3.10)

project(NeuralNetwork)

add_executable(main main.cpp)

add_subdirectory(NeuralNetworkLib)

target_link_libraries(main NeuralNetworkLib)

target_include_directories(main PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/NeuralNetworkLib"
                          "${PROJECT_SOURCE_DIR}/eigen-3.4.0"
                          )

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
