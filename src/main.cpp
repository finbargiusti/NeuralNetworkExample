#include <iostream>
#include "NeuralNetworkLib/NeuralNetwork.h"

int main() {
  NeuralNetwork nn (3);
  std::cout << nn.increment() << std::endl;
  return 0;
}
