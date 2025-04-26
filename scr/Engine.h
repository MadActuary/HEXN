#pragma once

#include "Model.h"
#include "Payoff.h"
#include <unordered_map>
#include <vector>
#include <random>

class Engine {

public:

    Engine(Model& model, const Payoff& payoff, int simulations = 10000);

    std::unordered_map<std::string, std::vector<double>> getCashflow(int moment = 1, int steps = 120, bool print = true, const std::string& fileName = "testCf.csv");

    void exportCashflowsCSV(const std::string& filename, std::unordered_map<std::string, std::vector<double>> cashflows);

private:

    Model& model;

    const Payoff& payoff;

    int simulations;
};
