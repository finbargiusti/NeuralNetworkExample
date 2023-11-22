#ifndef NEURALNETWORK_H
// helpful library for matmul etc.
#include "Eigen/Eigen"

#include <queue>

// Here we can define some types that will be used

// we will use floats for scalars
typedef float Scalar;

// we will use Eigen to define vectors, transposed vectors and matrices
typedef Eigen::RowVectorXf Vector;
typedef Eigen::VectorXf VectorT;
typedef Eigen::MatrixXf Matrix;

// the topology of the neural network will be defined as a series of integers,
// which determine the number of neurons in each layer, starting with the input
// layer. this can completely define the architecture of the neural network.

typedef std::vector<unsigned int> Topology;

typedef unsigned int Size;

typedef std::vector<Vector *> NetworkData;
typedef std::vector<Matrix *> NetworkWeights;

struct TrainingDatum {
  Vector input;
  Vector expected;
};

typedef std::vector<TrainingDatum> TrainingData;

class NeuralNetwork {
public:
  // Initialise the neural network give topology and learning rate
  NeuralNetwork(Topology topology);
  NeuralNetwork(Topology topology, NetworkWeights weights);

  // we need to add functions to read and write from disk.
  // (these will assume correctly formatted data so BE WARNED)

  // Fill the network weights with pseudo-random numbers
  void randomWeights();

  // Generate the output values of each neuron and to vector array.
  // NOTE: Will assume that the input is of the right size.
  Vector generate(Vector input);

  // train the network with an example
  // returns error vector
  Vector teach(Vector input, Vector expected, Scalar leanring_rate);

  // returns final error value
  void train(TrainingData data, Size epochs, Scalar bot_rate, Scalar top_rate, bool updateAfterEpoch = true);

  // update model weights with std. error.
  void updateWeights(Scalar learning_rate);

  // update error values based on expected result.
  // returns error on output layer
  void propogateError(Vector expected);

  void resetError();

  void initialiseVectors();

  // sigmoid activation function 

  Size num_layers;

  // how we will store the network in memory:
  
  Topology topology;

  // pre-activation function value for each layer's neurons
  NetworkData preActivation;
  // post-activation function (true) value for each layer's neurons
  NetworkData neurons;
  // calculated error for each layer's neurons
  NetworkData error;

  // weights (aka vector of matrices for matmul)
  NetworkWeights weights;

  //  useful for logging purposes only
  //{
  Size training_epochs;
  //}


};

Scalar smooth(Scalar x);

Scalar derivativeSmooth(Scalar x);

#endif

#define NEURALNETWORK_H


