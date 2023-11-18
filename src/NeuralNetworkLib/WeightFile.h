#include "NeuralNetwork.h"

// TODO: determine structure of weights written to disk
// - Need to write topology to file
// - Need to write weights to file (efficiently would be nice)
//
// to write, we can simply put the topology in the first line, and then
// the next toplogy,
// followed by the weights "between" the layers, in order.
// the weights can be written in raw bytes. delineated by an INFITITY, no delineator needed, since we will know the size at read time.


bool writeWeights(std::string filename, NetworkWeights weights);

// mutates array. (BE CAREFUL)
bool readWeights(std::string filename, NetworkWeights &weights); 

