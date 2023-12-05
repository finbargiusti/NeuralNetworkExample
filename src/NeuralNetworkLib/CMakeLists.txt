cmake_minimum_required(VERSION 3.10)

project(NeuralNetworkLib)

add_library(NeuralNetworkLib NeuralNetwork.cpp NeuralNetwork.h NetworkReflection.cpp NetworkReflection.h maths.cpp maths.h)

include_directories(NeuralNetworkLib PUBLIC
                          "${PROJECT_SOURCE_DIR}"
                          "${PROJECT_SOURCE_DIR}/eigen-3.4.0"
                          )
