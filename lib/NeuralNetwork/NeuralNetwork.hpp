//
//  NeuralNetwork.hpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 18/03/2023.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>
#include <fstream>
#include "Matrix.hpp"

// Reference
// back prop mathematical model: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
// neural network structure reference: https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/

using namespace std;

typedef vector<unsigned int> Topology;

inline double activate(double x)
{
    return tanh(x);
}

inline double activateDerivative(double x)
{
    return 1 - x * x;
}

class NeuralNetwork
{
private:
    Topology topo;
    double learningRate;

    Matrix *weights;     // store each node's weights
    Matrix *biases;      // biases of each layer
    Matrix *activations; // output of each layer
    Matrix *errors;      // store deltas of each layer

    // For input and ouput,
    // Use row vector

public:
    NeuralNetwork(Topology topo, double learningRate = 0.05);
    NeuralNetwork(ifstream &file);
    // NeuralNetwork(const NeuralNetwork &origin);
    ~NeuralNetwork();

    // Initialize the network, avoid code replication
    void initNetwork();

    // Foward pass the input
    void forwardPass(Matrix input);

    // Optimize the network using backwardPropagation
    void backwardPropagate(Matrix output);

    // Save network to file
    void saveNetwork(ofstream &file);

    // Train network
    Matrix train(Matrix input, Matrix output);

    // Test network
    Matrix test(Matrix input);
};

#endif /* NeuralNetwork_hpp */
