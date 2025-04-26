#pragma once

#include "Model.h"
#include "Payoff.h"
#include <unordered_map>
#include <vector>
#include <random>

class Engine {

public:

    Engine(Model& model, const Payoff& payoff, int simulations = 10000);

    std::unordered_map<std::string, std::vector<double>> getCashflow(int moment = 1, int steps = 120);


private:

    Model& model;

    const Payoff& payoff;

    int simulations;

};
