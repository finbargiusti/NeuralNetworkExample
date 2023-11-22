#include "NetworkReflection.h"

#include <fstream>

bool saveWeights(std::string filename, NetworkWeights &weights) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);

  if (!file.is_open()) {
    // opening error
    return false;
  }

  // we can write the weights. We can just write each byte in order,
  // since we know the size of each weight matrix.
  //
  // NOTE: no static type sizes, system independent.

  for (Size i = 0; i < weights.size(); i++) {
    Matrix *weightMatrix = weights[i];
    long rows = weightMatrix->rows();
    long cols = weightMatrix->cols();

    for (long row = 0; row < rows; row++) {
      for (long col = 0; col < cols; col++) {
        Scalar weight = (*weightMatrix)(row, col);
        file.write((char *)&weight, sizeof(Scalar));
      }
    }
  }

  file.close();

  return true;
};

NetworkWeights &readWeights(std::string filename, Topology &topology){
  std::ifstream file(filename, std::ios::in | std::ios::binary);

  NetworkWeights *weights = new NetworkWeights();

  if (!file.is_open()) {
    // opening error
    // HACK: hack?
    return *weights;
  }

  // Now we basically do the same thing, but in reverse.

  for (Size layer_index = 1; layer_index < topology.size(); layer_index++) {
    Size rows, cols;

    if (layer_index == topology.size() - 1) {
      rows = topology[layer_index - 1] + 1;
      cols = topology[layer_index];
    } else {
      rows = topology[layer_index - 1] + 1;
      cols = topology[layer_index] + 1;
    }

    Matrix *weightMatrix = new Matrix(rows, cols);

    for (Size row = 0; row < rows; row++) {
      for (Size col = 0; col < cols; col++) {
        Scalar weight;
        file.read((char *)&weight, sizeof(weight));
        (*weightMatrix)(row, col) = weight;
      }
    }

    weights->push_back(weightMatrix);
  }

  file.close();

  return *weights;
};

Topology &readTopology(std::string filename) {
  std::ifstream file (filename, std::ios::in);

  Topology *topology = new Topology();

  if (!file.is_open()) {
    // FIX: maybe an exception?
   
    // opening error
    return *topology;
  }

  Size n;

  while (file >> n) 
    topology->push_back(n);

  return *topology;
};

TrainingData readTrainingData(std::string filename, Topology topology) {
  std::ifstream file (filename, std::ios::in);

  TrainingData *trainingData = new TrainingData();

  if (!file.is_open()) {
    // FIX: maybe an exception?
   
    // opening error
    return *trainingData;
  }

  Size input_size, expected_size;

  input_size = topology[0];
  expected_size = topology.back();

  Scalar n;

  while (!file.eof()) {
    Vector input (input_size);
    Vector expected (expected_size);

    for (Size i = 0; i < input_size; i++) {
      file >> n;
      input(i) = n;
    }

    for (Size i = 0; i < expected_size; i++) {
      file >> n;
      expected(i) = n;
    }

    TrainingDatum datum = {input, expected};

    trainingData->push_back(datum);
  }

  return *trainingData;
};
