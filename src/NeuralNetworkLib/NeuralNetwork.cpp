#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
  counter = 0; 
}

NeuralNetwork::NeuralNetwork(int startValue) {
  counter = startValue; 
}

int NeuralNetwork::increment() {
  return ++counter;
}
