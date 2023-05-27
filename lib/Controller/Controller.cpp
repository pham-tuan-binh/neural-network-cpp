//
//  Controller.cpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 03/04/2023.
//

#include "Controller.hpp"

void Controller::render()
{
    econio_normalmode();
    econio_clrscr();
    printBanner();
    (this->*stateTransitions[currentState])();
    return;
}

void Controller::printBanner()
{
    std::ifstream f("./data/banner.txt");

    if (f.is_open())
    {
        econio_textcolor(COL_GREEN);
        econio_textbackground(COL_BLACK);
        std::cout << f.rdbuf();
        econio_textcolor(COL_RESET);
        econio_textbackground(COL_RESET);
        f.close();
    }
    return;
}

void Controller::generationMenu()
{
    cout << "[UP]: Move up [DOWN]: Move down [ENTER]: Select Mode" << endl;
    cout << "_______________________________________________________________________________________________________" << endl
         << endl;
    cout << "Please pick an option to kick start the neural network 900HP engine:" << endl;
    cout << (currentCursorPosition == 0 ? "[X]" : "[ ]") << " "
         << "1. Generate Neural Network" << endl;
    cout << (currentCursorPosition == 1 ? "[X]" : "[ ]") << " "
         << "2. Load Neural Network from file" << endl;

    econio_rawmode();
    int key = econio_getch();

    if (key == KEY_UP && currentCursorPosition > 0)
    {
        currentCursorPosition--;
    }
    if (key == KEY_DOWN && currentCursorPosition < 1)
    {
        currentCursorPosition++;
    }

    if (key == KEY_ENTER)
    {
        currentState = StateNumber(currentCursorPosition + 1);
    }
}

void Controller::controlMenu()
{
    cout << "[UP]: Move up [DOWN]: Move down [ENTER]: Select Mode" << endl;
    cout << "_______________________________________________________________________________________________________" << endl
         << endl;

    cout << "Vroom, vroom, the engine is started, pick an option below to play with it:" << endl;
    cout << (currentCursorPosition == 0 ? "[X]" : "[ ]") << " "
         << "1. Train Neural Network" << endl;
    cout << (currentCursorPosition == 1 ? "[X]" : "[ ]") << " "
         << "2. Test Neural Network" << endl;
    cout << (currentCursorPosition == 2 ? "[X]" : "[ ]") << " "
         << "3. Save Neural Network" << endl;
    cout << (currentCursorPosition == 3 ? "[X]" : "[ ]") << " "
         << "4. Exit Neural Network" << endl;
    cout << (currentCursorPosition == 4 ? "[X]" : "[ ]") << " "
         << "5. Return to previous menu" << endl;

    econio_rawmode();
    int key = econio_getch();

    if (key == KEY_UP && currentCursorPosition > 0)
    {
        currentCursorPosition--;
    }
    if (key == KEY_DOWN && currentCursorPosition < 4)
    {
        currentCursorPosition++;
    }

    if (key == KEY_ENTER)
    {
        currentState = StateNumber(currentCursorPosition + 4);
        if (currentCursorPosition == 4)
        {
            currentState = GEN_MENU;
            currentCursorPosition = 0;
        }
    }
}

void Controller::createNetwork()
{

    int numberOfHiddenLayers;
    Topology topology;

    cout << "[ENTER]: Per the instruction below" << endl;
    cout << "_______________________________________________________________________________________________________" << endl
         << endl;

    cout << "To create your network, you need to specify a topology." << endl
         << endl;

    if (cin.good())
    {
        cout << "Please enter the number of hidden layers: ";
    }
    else
    {
        econio_textcolor(COL_RED);
        econio_textbackground(COL_WHITE);
        cout << "Error: [Please enter INTERGER]" << endl;
        econio_textbackground(COL_RESET);
        econio_textcolor(COL_RESET);
        cout << "Please enter the number of hidden layers: ";
    }

    cin.clear();

    cin >> numberOfHiddenLayers;

    if (!cin.good())
    {
        return;
    }

    topology.push_back(1);
    for (int i = 0; i < numberOfHiddenLayers; i++)
    {
        int size = 0;
        cout << "Enter the size of hidden layer [" << i + 1 << "]: ";
        cin.clear();
        cin >> size;

        if (!cin.good())
        {
            return;
        }

        topology.push_back(size);
    }
    topology.push_back(1);

    // Create a neural network
    n = new NeuralNetwork(topology);

    currentState = CONTROL_MENU;
    return;
}

void Controller::trainNetwork()
{
    int max_iter;
    ofstream my_file, test_file;
    my_file.open("./data/data.csv");
    test_file.open("./data/test.csv");

    cout << "[ENTER]: Per the instruction below" << endl;
    cout << "_______________________________________________________________________________________________________" << endl
         << endl;

    if (cin.good())
    {
        cout << "To train the network, please enter the Epochs (Number of times to train the network): ";
    }
    else
    {
        econio_textcolor(COL_RED);
        econio_textbackground(COL_WHITE);
        cout << "Error: [Please enter INTERGER]" << endl;
        econio_textbackground(COL_RESET);
        econio_textcolor(COL_RESET);
        cout << "To train the network, please enter the Epochs (Number of times to train the network): ";
    }

    cin.clear();

    cin >> max_iter;

    if (!cin.good())
    {
        return;
    }

    cout << "Your network is being trained. It should take long if Epocks are large, you will be directed to previous menu if the training finishes." << endl;

    my_file << "count,x,y,y_guess,loss" << endl;

    for (int i = 1; i <= max_iter; ++i)
    {
        // generate (x, y) training data: y = sin^2(x)
        // Generaet input
        Matrix x = Matrix(1, 1) * 3.14;

        // Generate ouput
        Matrix y(x);
        y.applyFunction([](double v) -> double
                        { return sin(v) * sin(v); });

        // Train the network and get the result
        Matrix y_guess = n->train(x, y);

        // Output the result
        my_file << i << "," << x.local_array[0][0] << "," << y.local_array[0][0] << "," << y_guess.local_array[0][0] << "," << (y_guess.local_array[0][0] - y.local_array[0][0]) << endl;
    }

    my_file.close();

    test_file << "count,x,y,y_guess,loss" << endl;

    for (int i = 1; i <= 10000; ++i)
    {
        // generate (x, y) training data: y = sin^2(x)
        // Generaet input
        Matrix x = Matrix(1, 1) * 3.14;

        // Generate ouput
        Matrix y(x);
        y.applyFunction([](double v) -> double
                        { return sin(v) * sin(v); });

        // Train the network and get the result
        Matrix y_guess = n->test(x);

        // Output the result
        test_file << i << "," << x.local_array[0][0] << "," << y.local_array[0][0] << "," << y_guess.local_array[0][0] << "," << (y_guess.local_array[0][0] - y.local_array[0][0]) << endl;
    }

    test_file.close();

    currentState = CONTROL_MENU;
}

void Controller::testNetwork()
{
    cout << "To use the keys below, please follow the instructions first." << endl;
    cout << "[ESCAPE]: Escape this mode [ANY OTHER KEYS]: Clear current input" << endl;
    cout << "_______________________________________________________________________________________________________" << endl
         << endl;

    if (cin.good())
    {
        cout << "Type in the number (x) and the network will return (y)." << endl;
    }
    else
    {
        // Color setting
        econio_textcolor(COL_RED);
        econio_textbackground(COL_WHITE);

        cout << "Error: [Please enter DOUBLE]" << endl;

        econio_textbackground(COL_RESET);
        econio_textcolor(COL_RESET);
        cout << "Type in the number (x) and the network will return (y)." << endl;
    }

    cin.clear();

    Matrix x = Matrix(1, 1);
    Matrix y_guess = Matrix(1, 1);

    cin >> x.local_array[0][0];

    if (!cin.good())
    {
        return;
    }

    y_guess = n->test(x);
    cout << "Network result: " << y_guess.local_array[0][0] << " Expected result: " << sin(x.local_array[0][0]) * sin(x.local_array[0][0]) << endl;

    econio_rawmode();
    int key = econio_getch();
    if (key == KEY_ESCAPE)
    {
        currentState = CONTROL_MENU;
    }
}

void Controller::loadNetwork()
{
    ifstream neural_network;
    neural_network.open("./data/network.txt");
    n = new NeuralNetwork(neural_network);
    neural_network.close();
    currentState = CONTROL_MENU;
}

void Controller::saveNetwork()
{
    cout << "save network menu";
    ofstream save_file;
    save_file.open("./data/network.txt");
    n->saveNetwork(save_file);
    save_file.close();
    currentState = CONTROL_MENU;
}

void Controller::exit()
{
    renderFlag = false;

    econio_textcolor(COL_GREEN);
    econio_textbackground(COL_BLACK);
    cout << "Neural Network and State Machine along with tons of C++ nitty gritty. Thank you" << endl;
    econio_textbackground(COL_RESET);
    econio_textcolor(COL_RESET);
}
