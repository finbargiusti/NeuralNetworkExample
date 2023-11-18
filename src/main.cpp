#include <iostream>
#include "NeuralNetwork.h"
#include "NetworkReflection.h"
#include <fstream>

bool file_exists(std::string filename);

int main(int argc, char **argv) {
  // we get folder name as argv[1]
  
  if (argc < 2) {
    std::cout << "Please provide a folder name to save to." << std::endl;
    return 1;
  }

  std::string folder_name = argv[1];

  std::string topology_filename = folder_name + "/topology.txt";
  std::string weights_filename = folder_name + "/weights.bin";
  std::string training_data_filename = folder_name + "/training_data.txt";

  if (!file_exists(topology_filename)) {
    std::cout << "Topology file (topology.txt) does not exist." << std::endl;
    return 1;
  }

  Scalar learning_rate = 0.01;

  Topology topology = readTopology(topology_filename);

  NeuralNetwork *network;

  if (file_exists(weights_filename)) {
    // load weights
    NetworkWeights weights = readWeights(weights_filename, topology);
    network = new NeuralNetwork(topology, weights);
  } else {
    // create new network
    network = new NeuralNetwork(topology);
  }

  // main program loop

  while (!std::cin.eof()) {

    std::string command;

    std::cout << "> ";

    std::cin >> command; 

    if (command == "exit") {
      return 0;
    }

    if (command == "save") {
      saveWeights(weights_filename, network->weights);
      continue;
    }

    if (command == "train") {
      if (!file_exists(training_data_filename)) {
        std::cout << "Training data file (training_data.txt) does not exist." << std::endl;
        continue;
      }

      Size epochs;

      std::cin >> epochs;

      TrainingData training_data = readTrainingData(training_data_filename, topology);

      network->train(training_data, epochs, learning_rate);
      continue;
    }

    if (command == "test") {
      Size input_size = topology[0];

      std::cout << "Input size: (" << input_size << " number): " << std::endl;

      Vector input(input_size);

      Scalar val;

      for (Size i = 0; i < input_size; i++) {
        std::cin >> val;
        input.coeffRef(i) = val;
      }

      Vector output = network->generate(input);

      std::cout << "Output: " << output <<  std::endl;
      continue;
    }

    if (command == "rate") {
      std::cin >> learning_rate;
      continue;
    }

    std::cout << "Command not recognised." << std::endl;
  }
}
bool file_exists(std::string filename) {
  std::ifstream file (filename, std::ios::in);

  bool res = file.is_open();

  file.close();

  return res;
}
