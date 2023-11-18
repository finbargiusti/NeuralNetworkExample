#include <iostream>
#include "NeuralNetwork.h"

int main() {
  // basic XOR example
  
  std::vector<Vector> input;
  std::vector<Vector> expected;

  // add the following XOR examples:
  //
  // 0, 0 -> 0
  // 0, 1 -> 1
  // 1, 0 -> 1
  // 1, 1 -> 0
  
  Vector one = Vector::Ones(1);
  Vector zero = Vector::Zero(1);

  Vector zero_zero = Vector::Zero(2);
  Vector zero_one = Vector::Zero(2);
  zero_one.coeffRef(1) = 1.0;
  Vector one_zero = Vector::Zero(2);
  one_zero.coeffRef(0) = 1.0;
  Vector one_one = Vector::Ones(2);

  input.push_back(zero_zero);
  expected.push_back(zero);

  input.push_back(zero_one);
  expected.push_back(one);

  input.push_back(one_zero);
  expected.push_back(one);

  input.push_back(one_one);
  expected.push_back(zero);

  // define the topology of the network
  
  Topology topology;

  topology.push_back(2);
  topology.push_back(3);
  topology.push_back(3);
  topology.push_back(3);
  topology.push_back(1); // output layer
  
  NeuralNetwork network(topology, 0.05);

  network.randomWeights(0);

  network.train(input, expected, 100000, false);

  // test with inputs
  
  Scalar input1, input2;

  while (true) {
    // until break
    std::cout << "Enter two numbers (0-1): ";
    std::cin >> input1 >> input2;
    
    Vector cinput = Vector::Zero(2);

    // TODO: this is shit.
    cinput.coeffRef(0) = input1;
    cinput.coeffRef(1) = input2;

    std::cout << cinput;

    Vector output = network.generate(cinput);
    
    // print output
    //
    std::cout << "Output: " << output.coeffRef(0) << std::endl;
  }
}
