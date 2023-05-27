//
//  Controller.hpp
//  NeuralNetwork
//
//  Created by Pham Tuan Binh on 03/04/2023.
//

#ifndef Controller_hpp
#define Controller_hpp

#include <iostream>
#include "NeuralNetwork.hpp"
#include "econio.hpp"
#include <unistd.h>

using namespace std;

// The states of the machine are hard-coded

enum StateNumber
{
    GEN_MENU = 0,
    // Phase 1
    CREATE = 1,
    LOAD = 2,

    // Phase 2
    CONTROL_MENU = 3,
    TRAIN = 4,
    TEST = 5,
    SAVE = 6,
    EXIT = 7
};

class Controller
{
private:
    // internal neural network
    NeuralNetwork *n;

    StateNumber currentState;
    int currentCursorPosition;
    typedef void (Controller::*Action)();

    void printBanner();
    void generationMenu();
    void controlMenu();
    void createNetwork();
    void trainNetwork();
    void testNetwork();
    void loadNetwork();
    void saveNetwork();
    void exit();

    Action stateTransitions[8];

public:
    // Render flag to signal the system
    bool renderFlag;

    Controller()
    {
        stateTransitions[GEN_MENU] = &Controller::generationMenu;

        // Phase 1
        stateTransitions[CREATE] = &Controller::createNetwork;
        stateTransitions[LOAD] = &Controller::loadNetwork;

        // Phase 2
        stateTransitions[CONTROL_MENU] = &Controller::controlMenu;
        stateTransitions[TRAIN] = &Controller::trainNetwork;
        stateTransitions[TEST] = &Controller::testNetwork;
        stateTransitions[SAVE] = &Controller::saveNetwork;
        stateTransitions[EXIT] = &Controller::exit;

        currentState = GEN_MENU;
        currentCursorPosition = 0;
        renderFlag = true;
    };

    ~Controller()
    {
        delete n;
    };

    void render();
};

#endif /* Controller_hpp */
