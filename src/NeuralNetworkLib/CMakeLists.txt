cmake_minimum_required(VERSION 3.10)

project(NeuralNetworkLib)

add_library(NeuralNetworkLib NeuralNetwork.cpp NeuralNetwork.h)

include_directories(NeualNetworkLib PUBLIC
                          "${PROJECT_SOURCE_DIR}/eigen-3.4.0"
                          )
