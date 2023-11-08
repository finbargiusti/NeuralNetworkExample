// helpful library for matmul etc.
#include "Eigen/Eigen"

// Here we can define some types that will be used

// we will use floats for scalars
typedef float Scalar;

// we will use Eigen to define vectors, transposed vectors and matrices
typedef Eigen::RowVectorXf Vector;
typedef Eigen::VectorXf VectorT;
typedef Eigen::MatrixXf Matrix;

// the topology of the neural network will be defined as a series of integers, 
// which determine the number of neurons in each layer, starting with the input layer.
// this can completely define the architecture of the neural network.

typedef std::vector<unsigned int> Topology;

typedef unsigned int size;

typedef std::vector<Vector*> NetworkData;
typedef std::vector<Matrix*> NetworkWeights;

class NeuralNetwork {
public:

  // Initialise the neural network give topology and learning rate
  NeuralNetwork(Topology topology, Scalar learning_rate = 0.01);

  // we need to add functions to read and write from disk.
  // (these will assume correctly formatted data so BE WARNED)
  
  // Fill the network weights with pseudo-random numbers
  void fillRandom(int seed);
  
  // Read the network weights from a file
  // Returns false on any error
  bool read(std::string filename);

  // Write the network weights to a file
  // Returns false on any error
  bool write(std::string filename);

private:

  size num_layers;

  // how we will store the network in memory:
  
  // pre-activation function value for each layer's neurons
  NetworkData preActivation;
  // post-activation function (true) value for each layer's neurons
  NetworkData postActivation;
  // calculated error for each layer's neurons
  NetworkData error;

  // weights (aka vector of matrices for matmul)
  NetworkWeights weights;

//  useful for logging purposes only
//{
  size num_neurons;
  size training_epochs = 0;
//}
};
