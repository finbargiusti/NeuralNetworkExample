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
// which determine the number of neurons in each layer, starting with the input
// layer. this can completely define the architecture of the neural network.

typedef std::vector<unsigned int> Topology;

typedef unsigned int Size;

typedef std::vector<Vector *> NetworkData;
typedef std::vector<Matrix *> NetworkWeights;

class NeuralNetwork {
public:
  // Initialise the neural network give topology and learning rate
  NeuralNetwork(Topology topology, Scalar learning_rate = 0.01);

  // we need to add functions to read and write from disk.
  // (these will assume correctly formatted data so BE WARNED)

  // Fill the network weights with pseudo-random numbers
  void randomWeights(int seed);

  // Read the network weights from a file
  // Returns false on any error
  bool readWeights(std::string filename);

  // Write the network weights to a file
  // Returns false on any error
  bool writeWeights(std::string filename);


  // Generate the output values of each neuron and to vector array.
  // NOTE: Will assume that the input is of the right size.
  Vector generate(Vector input);

  // train the network with an example
  void teach(Vector input, Vector expected);

  void train(std::vector<Vector> input, std::vector<Vector> expected, Size epochs, bool updateAfterEpoch = true);

  // update model weights with std. error.
  void updateWeights();

private:

  // update error values based on expected result.
  void propogateError(Vector expected);

  void resetError();

  // sigmoid activation function 
  const static Scalar sigmoid(Scalar x);

  const static Scalar derivativeSigmoid(Scalar x);

  Size num_layers;
  Scalar learning_rate;

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
  Size num_neurons;
  Size training_epochs = 0;
  //}
};
