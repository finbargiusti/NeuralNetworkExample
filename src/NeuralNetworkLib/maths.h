#ifndef MATHS_H

#include "NeuralNetwork.h"

std::function<Scalar(Scalar)> unaryActivation(ActivationFunction a);

std::function<Scalar(Scalar)> unaryActivationDerivative(ActivationFunction a);

Vector activation(Vector v, ActivationFunction a);

Vector activationDerivative(Vector v, ActivationFunction a);

Scalar sabs(Scalar x);


Scalar dyn_learning_rate(Scalar top_rate, Scalar bot_rate, Size cycle_length,
                         Scalar decay_rate, Size epoch);



#define MATHS_H

#endif
