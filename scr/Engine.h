#pragma once
#include "Model.h"
#include "Payoff.h"
#include <unordered_map>
#include <vector>
#include <string>

class Engine {
public:
    Engine(Model& model, const Payoff& payoff, int simulations);

    std::unordered_map<std::string, std::vector<double>>
        getCashflow(int moment, int steps, bool print = false, const std::string& fileName = "");

private:
    Model& model;
    const Payoff& payoff;
    int simulations;

    void exportCashflowsCSV(const std::string& filename,
        const std::unordered_map<std::string, std::vector<double>>& cashflows);
};

