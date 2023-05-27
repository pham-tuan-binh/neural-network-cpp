//
//  main.cpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 18/03/2023.
//

#include <iostream>
#include <time.h>
#include <vector>
#include <math.h>
#include <fstream>
#include <bitset>
#include <cstdlib>

// Custom Library written by me
#include "Matrix.hpp"        // Matrix is the backbone of the neural network
#include "NeuralNetwork.hpp" // Neural network utilizes matrix math to operate.
#include "Controller.hpp"    // Controller wraps everything in a nice to use UI

using namespace std;

// Thanks to an insane amount of abstraction I was able to reduce the main to a super easy to understand format
int main()
{
    try
    {
        // Initialize the controller
        Controller a;

        // While the render flag is on, enable the render
        // Sometimes it is off if a
        while (a.renderFlag)
        {
            a.render();
        }
    }
    catch (domain_error e)
    {
        cout << e.what();
    }
    return 0;
}
