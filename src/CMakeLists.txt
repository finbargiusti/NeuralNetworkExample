cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(NeuralNetwork)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp)

add_subdirectory(NeuralNetworkLib)

target_link_libraries(main NeuralNetworkLib)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
  "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
)

target_include_directories(main PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/NeuralNetworkLib"
                          "${PROJECT_SOURCE_DIR}/NeuralNetworkLib/eigen-3.4.0"
                          )
