#include "NeuralNetwork.h"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <functional>
#include <thread>

NeuralNetwork::NeuralNetwork(Topology topology) {
  // Now we must initialise the network givin a topology;

  this->topology = topology; // for later reference.

  initialiseVectors();
  // initialise weights
  randomWeights();
}

NeuralNetwork::NeuralNetwork(Topology topology, NetworkWeights weights) {
  // Now we must initialise the network givin a topology;

  this->topology = topology; // for later reference.
  this->weights = weights;

  initialiseVectors();
  // initialise weights
}

void NeuralNetwork::initialiseVectors() {
  for (Size layer_index = 0; layer_index < topology.size(); layer_index++) {
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

void NeuralNetwork::randomWeights() {
  for (Size layer_index = 1; layer_index < neurons.size(); layer_index++) {
    Size n, m;
    n = neurons[layer_index - 1]->size();
    m = neurons[layer_index]->size();
    if (layer_index < neurons.size() - 1)
      m--; // remove bias neuron from next layer

    weights.push_back(new Matrix(n, m));

    // set eigien matrix coefficients to random values
    weights.back()->setRandom(n, m);
  }
}

Scalar smooth(Scalar x) { return tanh(x); }

Scalar derivativeSmooth(Scalar x) {
  return 1.0 - (tanh(x) * tanh(x)); // derivative of tanh
}

Vector NeuralNetwork::generate(Vector input) {

  // set input layer to input (excluding bias)

  neurons.front()->block(0, 0, 1, input.size()) = input;

  for (Size layer_index = 1; layer_index < neurons.size(); layer_index++) {

    Size num_to_update = neurons[layer_index]->size() - 1;
    
    if (layer_index == neurons.size()- 1)
      num_to_update++;

    // calculate preActivation for this layer (excluding bias)
    preActivation[layer_index]->block(0, 0, 1, num_to_update) =
        (*neurons[layer_index - 1]) * (*weights[layer_index - 1]);

    neurons[layer_index]->block(0, 0, 1, num_to_update) = 
      preActivation[layer_index]->block(0, 0, 1, num_to_update).unaryExpr(&smooth);
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

  // calculate error for hidden layers
  for (Size layer_index = error.size() - 2; layer_index >= 0; layer_index--) {
    // calculate error for hidden layers
    
    Size erring_neurons = error[layer_index + 1]->size() - 1;

    if (layer_index == error.size() - 2)
      erring_neurons++;
  
    (*error[layer_index]) = 
      error[layer_index + 1]->block(0, 0, 1, erring_neurons) * weights[layer_index + 1]->transpose();;
    (*error[layer_index]) = 
      error[layer_index]->cwiseProduct(preActivation[layer_index + 1]->unaryExpr(&derivativeSmooth));

    if (error[layer_index]->hasNaN()) {
      std::cout << "NaN in error!" << '\n' << *error[layer_index] << '\n';
      throw std::invalid_argument("Nan in error");
    }

    if (layer_index == 0)
      return; // HACK: WTF?
  }
}

void NeuralNetwork::resetError() {
  for (Size layer_index = 0; layer_index < error.size(); layer_index++) {
    error[layer_index]->setZero();
  }
}

void NeuralNetwork::updateWeights(Scalar learning_rate) {

  // update weights based on error and learning rate
  for (Size layer_index = 0; layer_index < weights.size(); layer_index++) {

    Matrix *layer_weights = weights[layer_index];

    for (int col = 0; col < layer_weights->cols(); col++) {
      for (int row = 0; row < layer_weights->rows(); row++) {
        Scalar delta = learning_rate * error[layer_index]->coeffRef(col) *
                       neurons[layer_index]->coeffRef(row);
        layer_weights->coeffRef(row, col) -= delta;
      }
    }

  }
}

Scalar sabs(Scalar x) { return x > 0 ? x : -x; }

Vector NeuralNetwork::teach(Vector input, Vector expected,
                            Scalar learning_rate) {
  // generate output
  generate(input);
  // propogate error
  propogateError(expected);
  Vector score = *error.back();
  // update weights
  updateWeights(learning_rate);
  // reset error
  resetError();

  return score;
}

// TODO: make this configurable?
#define DECAY_RATE 0.001
#define CYCLE_LENGTH 1000

Scalar dyn_learning_rate(Scalar top_rate, Scalar bot_rate, Size epoch) {
  return (top_rate - ((top_rate - bot_rate) * ((epoch % CYCLE_LENGTH) / CYCLE_LENGTH))) / 
          (1 + DECAY_RATE * epoch);
}

void NeuralNetwork::train(TrainingData data, Size epochs, Scalar top_rate,
                          Scalar bot_rate, std::function<int(Size epoch, Scalar error, Scalar learning_rate)> trainStatisticHook) {

  Scalar first_error, last_error;

  // train the network with a set of examples
  for (Size epoch = 0; epoch < epochs; epoch++) {

    Scalar res_error = 0.0;

    // calculate dynamic learning rate

    Scalar dynamic_learning_rate = dyn_learning_rate(top_rate, bot_rate, epoch);

    for (Size i = 0; i < data.size(); i++) {
      Vector score =
          teach(data[i].input, data[i].expected, dynamic_learning_rate);
      res_error += score.unaryExpr(&sabs).sum();
    }

    res_error /= data.size();

    if (epoch == 0)
      first_error = res_error;

    if (epoch == epochs - 1)
      last_error = res_error;

    trainStatisticHook(epoch, res_error, dynamic_learning_rate);
  }


  training_epochs += epochs;
}
