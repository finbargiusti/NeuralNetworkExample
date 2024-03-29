#include "maths.h"

std::function<Scalar(Scalar)> unaryActivation(ActivationFunction a) {
  switch (a) {
  case ActivationFunction::SIGMOID:
    return [](Scalar x) -> Scalar { return 1.0 / (1.0 + exp(-x)); };
  case ActivationFunction::TANH:
    return [](Scalar x) -> Scalar { return tanh(x); };
  case ActivationFunction::BINARY:
    return [](Scalar x) -> Scalar {
      if (x > 0)
        return 1.0;
      else
        return 0.0;
    };
  default:
    // none
    return [](Scalar x) -> Scalar { return x; };
  }
}

std::function<Scalar(Scalar)> unaryActivationDerivative(ActivationFunction a) {
  switch (a) {
  case ActivationFunction::SIGMOID:
    return [](Scalar x) -> Scalar {
      return (1.0 / (1.0 + exp(-x))) * (1 - 1.0 / (1.0 + exp(-x)));
    };
  case ActivationFunction::TANH:
    return [](Scalar x) -> Scalar { return 1 - tanh(x) * tanh(x); };
  default:
    // none or binary
    return [](Scalar x) -> Scalar { return 1.0; };
  }
}


  Scalar sabs(Scalar x) { return x > 0 ? x : -x; }

  Scalar dyn_learning_rate(Scalar top_rate, Scalar bot_rate, Size cycle_length,
                           Scalar decay_rate, Size epoch) {
    return (top_rate - ((top_rate - bot_rate) * ((float)(epoch % cycle_length) / cycle_length))) /
           (1 + decay_rate * epoch);
  }
