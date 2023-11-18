#include "NeuralNetwork.h"
#include <iostream>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(Topology topology, Scalar learning_rate) {
  // Now we must initialise the network givin a topology;

  this->topology = topology; // for later reference.
  this->learning_rate = learning_rate;

  num_neurons = 0;

  for (Size layer_index = 0; layer_index < topology.size(); layer_index++) {

    num_neurons += topology[layer_index];

    // for all layers but the output, we want to have an extra neuron value to
    // serve as the bias.
    Size layer_size = layer_index < topology.size() - 1
                          ? topology[layer_index] + 1
                          : topology[layer_index];

    neurons.push_back(new Vector(layer_size));
    preActivation.push_back(new Vector(layer_size));
    if (layer_index != 0)
      error.push_back(new Vector(layer_size));

    // Bias coefficient is 1 always!!!
    if (layer_index != topology.size() - 1) {
      neurons.back()->coeffRef(layer_size - 1) = 1.0;
      preActivation.back()->coeffRef(layer_size - 1) = 1.0;
    }
  }
}

void NeuralNetwork::randomWeights(int seed) {
  for (Size layer_index = 1; layer_index < neurons.size(); layer_index++) {
    Size n, m;
    n = neurons[layer_index - 1]->size();
    m = neurons[layer_index]->size();
    weights.push_back(new Matrix(n, m));

    // set eigien matrix coefficients to random values
    weights.back()->setRandom(n, m);
  }
}

bool NeuralNetwork::readWeights(std::string filename) {
  // TODO:
  // Implement
  return false;
}

bool NeuralNetwork::writeWeights(std::string filename) {
  // TODO:
  // Implement
  return false;
}

const Scalar NeuralNetwork::sigmoid(Scalar x) { return 1.0 / (1.0 + exp(-x)); }

const Scalar NeuralNetwork::derivativeSigmoid(Scalar x) {
  return sigmoid(x) * pow((1.0 - sigmoid(x)), 2);
}

Vector NeuralNetwork::generate(Vector input) {

  // set input layer to input (excluding bias)

  neurons.front()->block(0, 0, 1, input.size()) = input;

  for (Size layer_index = 1; layer_index < neurons.size(); layer_index++) {

    // calculate preActivation
    preActivation[layer_index]->noalias() =
        (*neurons[layer_index - 1]) * (*weights[layer_index - 1]);

    (*neurons[layer_index]) = preActivation[layer_index]->unaryExpr(&sigmoid);


  }
  // return output layer
  return *neurons.back();
}

void NeuralNetwork::propogateError(Vector expected) {
  // NOTE: here we do not update the error, but sum it so that we can update
  // weights after a btch of training examples. We will set the number of
  // examples until weights are updated in the train function.

  // calculate error for output layer, taking the sigmoid derivative into
  // account
  (*error.back()) = (*neurons.back()) - expected;

  std::cout << "Error: " << (*error.back())(0) << '\n';

  // calculate error for hidden layers
  for (Size layer_index = error.size() - 2; layer_index >= 0; layer_index--) {
    // calculate error for hidden layers
    error[layer_index]->noalias() =
        (*error[layer_index + 1]) * (*weights[layer_index + 1]).transpose();

    if (error[layer_index]->hasNaN()) {

      std::cout << "NaN in error!" << '\n' << *error[layer_index] << '\n';
      throw std::invalid_argument("Nan in error");
    }

    // if (error[layer_index]->coeffRef(0) ==
    // std::numeric_limits<Scalar>::infinity() ||
    // error[layer_index]->coeffRef(0) ==
    // -std::numeric_limits<Scalar>::infinity())
    // {
    //   std::cout << "Inf in error! 2" << '\n' << *error[layer_index] << '\n'
    //   << *preActivation[layer_index] << '\n' << product << '\n' <<
    //   *weights[layer_index] << '\n' << *error[layer_index + 1] << '\n'; throw
    //   std::invalid_argument("Inf in error");
    // };

    if (layer_index == 0)
      return; // HACK: WTF?
  }
}

void NeuralNetwork::resetError() {
  for (Size layer_index = 0; layer_index < error.size(); layer_index++) {
    error[layer_index]->setZero();
  }
}

void NeuralNetwork::updateWeights() {
  // update weights based on error and learning rate
  for (Size layer_index = 0; layer_index < weights.size(); layer_index++) {

    for (Size c = 0; c < weights[layer_index]->cols(); c++) {
      for (Size r = 0; r < weights[layer_index]->rows(); r++) {
        weights[layer_index]->coeffRef(r, c) -=
            learning_rate * (*error[layer_index])(c) *
            (preActivation[layer_index + 1]->unaryExpr(&derivativeSigmoid))(c) *
            (*neurons[layer_index])(r);
      }
    }

    if (weights[layer_index]->hasNaN()) {

      std::cout << "NaN in weights!" << '\n' << *error[layer_index] << '\n';
      throw std::invalid_argument("Nan in weights");
    };
  }
}

void NeuralNetwork::teach(Vector input, Vector expected) {
  // generate output
  generate(input);
  // propogate error
  propogateError(expected);
  // update weights
  updateWeights();
  // reset error
  resetError();
}

void NeuralNetwork::train(std::vector<Vector> input,
                          std::vector<Vector> expected, Size epochs,
                          bool updateAfterEpoch) {

  // train the network with a set of examples
  for (Size epoch = 0; epoch < epochs; epoch++) {
    for (Size i = 0; i < input.size(); i++) {
      teach(input[i], expected[i]);
    }
    // print error to screen
    // std::cout << "Epoch " << training_epochs + epoch << " complete." << '\n';
  }

  // print out all weight matrices for manual inspection.
  for (Size layer_index = 0; layer_index < weights.size(); layer_index++) {
    std::cout << "Weights for layer " << layer_index << '\n'
              << *weights[layer_index] << '\n';
  } 

  training_epochs += epochs;
}
