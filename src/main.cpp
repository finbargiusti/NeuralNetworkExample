#include "NeuralNetwork.h"
#include "NetworkReflection.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>


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

  std::vector<std::string> tokens;
  std::string input;

  while (!std::cin.eof()) {

    tokens.clear();

    std::cout << "> ";

    getline(std::cin, input);

    std::istringstream input_stream(input);

    for (std::string command; std::getline(input_stream, command, ' '); tokens.push_back(command));

    if (tokens[0] == "exit") {
      return 0;
    }

    if (tokens[0] == "save") {
      if (saveWeights(weights_filename, network->weights)) {
        print_info("Weights saved to file " + weights_filename + ".");
      } else {
        print_error("Failed to save weights to file " + weights_filename + ".");
      }
      continue;
    }

    if (tokens[0] == "train") {

      if (tokens.size() < 3) {
        print_error("Usage: train <epochs> <statistics file (relative to network dir)>");
        continue;
      }

      std::string statistics_filename = folder_name + "/" + tokens[2];

      if (!file_exists(training_data_filename)) {
        print_error("Training data file (training_data.txt) does not exist.");
        continue;
      }

      if (file_exists(statistics_filename)) {
        print_error("Statistics file already exists. Please delete it or choose a different name.");
        continue;
      }

      std::ofstream statistics_file(statistics_filename, std::ios::out);

      if (!statistics_file.is_open()) {
        print_error("Failed to open statistics file for writing.");
        continue;
      }

      statistics_file << "epoch,error,learning_rate" << std::endl;

      Size epochs = std::stoi(tokens[1]);

      print_info("Training network for " + tokens[1] + "  epochs.");

      TrainingData training_data = readTrainingData(training_data_filename, topology);

      Size start_time = std::clock();

      Scalar start_error;
      Scalar end_error;

      network->setTrainStatisticHook([&start_error, &end_error, &statistics_file](Size epoch, Scalar error, Scalar learning_rate) -> int {
        if (epoch == 0) start_error = error;
        end_error = error;
        std::cout << "epoch " << epoch << " average error: " << error
                  << " rate: " << learning_rate << "\t\r" << std::flush;

        statistics_file << epoch << "," << error << "," << learning_rate << std::endl;
        return 1;
      });

      network->train(training_data, epochs, top_rate, bot_rate);
    // print average error for last epoch

      std::cout << "\n Error went from " << start_error << " to " << end_error
              << " over " << epochs << " epochs, with learning rate [" << bot_rate << " - " << top_rate << "]."
              << std::endl;

      statistics_file.close();

      Size tot_time = std::clock() - start_time;

      Scalar ms_time = tot_time / ((Scalar) CLOCKS_PER_SEC * 1000);

      Scalar average_time = ms_time / (epochs);

      print_info("Training complete.");

      print_info("Training took " + std::to_string(average_time) + " ms per epoch on average.");

      continue;
    }

    if (tokens[0] == "generate") {


      Size input_size = topology[0];

      if (tokens.size() < 1 + input_size) {
        print_error("Usage: generate <inputs (size = " + std::to_string(input_size) + ")>");
        continue;
      }

      Vector input(input_size);

      Scalar val;

      for (Size i = 0; i < input_size; i++) {
        val = std::stof(tokens[1 + i]);
        input.coeffRef(i) = val;
      }

      Vector output = network->generate(input);

      std::cout << "Output: " << output <<  std::endl;
      continue;
    }

    if (tokens[0] == "rate") {
      if (tokens.size() < 3) {
        print_error("Usage: rate <top rate> <bot rate>");
        continue;
      }
      top_rate = std::stof(tokens[1]);
      bot_rate =  std::stof(tokens[2]);
      continue;
    }

    print_error( "Command not recognised: " + tokens[0] + ".");
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



