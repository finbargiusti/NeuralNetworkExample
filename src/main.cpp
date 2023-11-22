#include "NeuralNetwork.h"
#include "NetworkReflection.h"

#include <iostream>
#include <fstream>
#include <ctime>


bool file_exists(std::string filename);
void print_error(std::string msg);
void print_info(std::string msg);

int main(int argc, char **argv) {
  // we get folder name as argv[1]
  
  std::cout << "Welcome to the Neural Network Console." << "\n";
  std::cout << "By Finbar Giusti (21372821)" << std::endl;
  std::cout << "Licensed under WTFPL (http://www.wtfpl.net/txt/copying)" << std::endl;
  
  if (argc < 2) {
    print_error( "Usage: " + std::string(argv[0]) + " <network folder>");
    return 1;
  }

  std::string folder_name = argv[1];

  std::string topology_filename = folder_name + "/topology.txt";
  std::string weights_filename = folder_name + "/weights.bin";
  std::string training_data_filename = folder_name + "/training_data.txt";

  if (!file_exists(topology_filename)) {
    print_error("Topology file (topology.txt) does not exist.");
    return 1;
  }

  Scalar top_rate = 0.01;
  Scalar bot_rate = 0.0001;

  Topology topology = readTopology(topology_filename);

  NeuralNetwork *network;

  if (file_exists(weights_filename)) { 
    print_info("Loading weights from file " + weights_filename + "...");
    // load weights
    NetworkWeights weights = readWeights(weights_filename, topology);
    print_info("Weights loaded.");
    network = new NeuralNetwork(topology, weights);
  } else {
    print_info("No weights file found. Creating new network with random weights...");
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
      if (saveWeights(weights_filename, network->weights)) {
        print_info("Weights saved to file " + weights_filename + ".");
      } else {
        print_error("Failed to save weights to file " + weights_filename + ".");
      }
      continue;
    }

    if (command == "train") {
      if (!file_exists(training_data_filename)) {
        print_error("Training data file (training_data.txt) does not exist.");
        continue;
      }

      Size epochs;

      std::cin >> epochs;

      print_info("Training network for " + std::to_string(epochs) + "  epochs.");

      TrainingData training_data = readTrainingData(training_data_filename, topology);

      Size start_time = std::clock();

      network->train(training_data, epochs, top_rate, bot_rate);

      Size tot_time = std::clock() - start_time;

      Scalar ms_time = tot_time / ((Scalar) CLOCKS_PER_SEC * 1000);

      Scalar average_time = ms_time / (epochs);

      print_info("Training complete.");

      print_info("Training took " + std::to_string(average_time) + " ms per epoch on average.");

      continue;
    }

    if (command == "test") {
      Size input_size = topology[0];

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
      std::cin >> bot_rate >> top_rate;
      continue;
    }

    print_error( "Command not recognised." );
  }
}
bool file_exists(std::string filename) {
  std::ifstream file (filename, std::ios::in);

  bool res = file.is_open();

  file.close();

  return res;
}

void print_error(std::string msg) {
  // print error message, with some nice ANSI colors, if supported
  std::cout << "\033[1;31m"<< "[ERROR] " << msg << "\033[0m" << std::endl;
}

void print_info(std::string msg) {
  // print info message, with some nice ANSI colors, if supported
  std::cout << "\033[1;32m"<< "[INFO] " << msg << "\033[0m" << std::endl;
}



