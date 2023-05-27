//
//  NeuralNetwork.cpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 18/03/2023.
//

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(Topology topo, double learningRate)
{
    this->topo = topo;
    this->learningRate = learningRate;
    initNetwork();
}

/*NeuralNetwork::NeuralNetwork(const NeuralNetwork &original)
{
    topo = original.topo;
    learningRate = original.learningRate;

    weights = new Matrix[topo.size() - 1];
    biases = new Matrix[topo.size() - 1];
    errors = new Matrix[topo.size() - 1];

    activations = new Matrix[topo.size()];

    for (unsigned int i = 0; i < topo.size() - 1; i++)
    {
        weights[i] = original.weights[i];
        biases[i] = original.biases[i];
        errors[i] = original.errors[i];
    }

    for (unsigned int i = 0; i < topo.size(); i++)
    {
        activations[i] = original.activations[i];
    }
}*/

NeuralNetwork::~NeuralNetwork()
{
    delete[] weights;
    delete[] errors;
    delete[] activations;
    delete[] biases;
}

NeuralNetwork::NeuralNetwork(ifstream &file)
{
    int size;
    file >> size; // read the first value from the file
    file >> learningRate;

    topo.resize(size);
    for (int i = 0; i < size; i++)
    {
        file >> topo[i];
    }

    initNetwork();

    for (int i = 0; i < topo.size() - 1; i++)
    {
        weights[i] = Matrix(file);
    }

    for (int i = 0; i < topo.size() - 1; i++)
    {
        biases[i] = Matrix(file);
    }
}

void NeuralNetwork::initNetwork()
{
    // Everything has topo - 1 size, because we don't need weights,... for the input layer
    weights = new Matrix[topo.size() - 1];
    biases = new Matrix[topo.size() - 1];
    errors = new Matrix[topo.size() - 1];

    // Activations store the values of outputs of each node, hence the full size
    activations = new Matrix[topo.size()];

    // Randomize the values of weights and biases
    for (unsigned int i = 0; i < topo.size() - 1; i++)
    {
        weights[i] = Matrix(topo[i], topo[i + 1], true);
        biases[i] = Matrix(1, topo[i + 1], true);
    }
}

void NeuralNetwork::forwardPass(Matrix input)
{
    // Set input to activation
    activations[0] = input;

    // Pass through the whole network
    for (unsigned int i = 0; i < topo.size() - 1; i++)
    {
        // Initial values by multiplying inputs with weights matrix
        Matrix dot = activations[i] * weights[i];

        // Add the biases
        Matrix hidden = dot + biases[i];

        // Activate the outputs (to introduce non-linear)
        hidden.applyFunction(activate);

        // Save the outputs
        activations[i + 1] = hidden;
    }
}

void NeuralNetwork::saveNetwork(ofstream &file)
{
    file << topo.size() << " " << learningRate << endl;

    for (int i = 0; i < topo.size(); i++)
    {
        file << topo[i] << " ";
    }

    file << endl;

    for (int i = 0; i < topo.size() - 1; i++)
    {
        weights[i].saveMatrix(file);
    }

    for (int i = 0; i < topo.size() - 1; i++)
    {
        biases[i].saveMatrix(file);
    }

    return;
};

void NeuralNetwork::backwardPropagate(Matrix output)
{
    // Calculate the deltas
    errors[topo.size() - 2] = activations[topo.size() - 1] - output;

    // Backward traverse the network
    for (unsigned int i = topo.size() - 2; i > 0; i--)
    {
        // This part is totally from
        // Matt Mazur's mathematical model: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        // Basically, we generate a gradient of errors based on the speed of change of outputs based on the change of weights (or d(act)/d(weights))

        Matrix delta = errors[i] * weights[i].transpose();

        // Since we saved the activated values in activate, we need to deriv it back to calculate the errors.
        Matrix derivative = activations[i];
        derivative.applyFunction(activateDerivative);

        // Formula from Matt Mazur's
        errors[i - 1] = delta.hadamard(derivative);
    }

    // Update the values
    // If we know how much an output change if we change the weights
    // We can update the weights so that the output match perfectly with the desired output
    // We use learningRate to control how much we want it to match
    for (unsigned int i = 0; i < topo.size() - 1; i++)
    {
        Matrix delta = activations[i].transpose() * errors[i];
        weights[i] = weights[i] - (delta * learningRate);
        biases[i] = biases[i] - (errors[i] * learningRate);
    }

    return;
}

Matrix NeuralNetwork::test(Matrix input)
{
    // To test the network, we just need to foward pass the input
    forwardPass(input);

    return activations[topo.size() - 1];
}

Matrix NeuralNetwork::train(Matrix input, Matrix output)
{
    forwardPass(input);
    backwardPropagate(output);

    return activations[topo.size() - 1];
}
