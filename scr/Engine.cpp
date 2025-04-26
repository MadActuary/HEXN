#include "Engine.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "formatDouble.h"

Engine::Engine(Model& model, const Payoff& payoff, int simulations)
    : model(model), payoff(payoff), simulations(simulations) {}

void Engine::exportCashflowsCSV(const std::string& filename, std::unordered_map<std::string, std::vector<double>> cashflows) {
    // 2) collect and sort the state names (excluding "Total"), then add "Total" last
    std::vector<std::string> states;

    for (auto& kv : cashflows) {
        if (kv.first != "Total")
            states.push_back(kv.first);
    }
    std::sort(states.begin(), states.end());
    states.push_back("Total");

    // 3) open the output file
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        throw std::runtime_error("Engine::exportCashflowsCSV – cannot open file: " + filename);
    }

    // 4) write header row
    for (size_t i = 0; i < states.size(); ++i) {
        ofs << states[i]
            << (i + 1 < states.size() ? ";" : "\n");
    }

    // 5) write one row per t = 0..steps
    for (int t = 0; t <= cashflows.begin()->second.size() - 1; ++t) {
        for (size_t i = 0; i < states.size(); ++i) {
            const auto& st = states[i];
            ofs << formatDouble(cashflows[st][t])
                << (i + 1 < states.size() ? ";" : "\n");
        }
    }

    ofs.close();
}


std::unordered_map<std::string, std::vector<double>>

Engine::getCashflow(int moment, int steps, bool print, const std::string& fileName) {
    // 1) Prepare accumulators
    //   sums[state][t] = sum of payoffs at time t for that state
    std::unordered_map<std::string, std::vector<double>> sums;
    std::vector<double> totalSums(steps + 1, 0.0);

    // 2) Remember original model‐state
    std::string origState = model.getCurrentState();
    size_t      origAge = model.getAge();
    size_t      origInState = model.getDurationInState();
    size_t      origSinceB = model.getDurationSinceB();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // 3) Monte Carlo
    for (int sim = 0; sim < simulations; ++sim) {
        model.reset(origState, origAge, origInState, origSinceB);

        // t = 0
        {
            auto s = model.getCurrentState();
            auto d = model.getDurationInState();
            double pv = payoff.evaluate(s, d);
            auto& vec = sums
                .emplace(s, std::vector<double>(steps + 1, 0.0))
                .first->second;
            vec[0] += pv;
            totalSums[0] += pv;
        }

        // t = 1 … steps
        for (int t = 1; t <= steps; ++t) {
            double u = dis(gen);
            model.step(u);

            auto s = model.getCurrentState();
            auto d = model.getDurationInState();
            double pv = payoff.evaluate(s, d);

            auto& vec = sums
                .emplace(s, std::vector<double>(steps + 1, 0.0))
                .first->second;
            vec[t] += pv;
            totalSums[t] += pv;
        }
    }

    // 4) Build averages
    std::unordered_map<std::string, std::vector<double>> cashflows;
    for (auto& entry : sums) {
        const auto& state = entry.first;
        auto& sumV = entry.second;
        std::vector<double> avg(steps + 1);
        for (int t = 0; t <= steps; ++t) {
            avg[t] = sumV[t] / simulations;
        }
        cashflows[state] = std::move(avg);
    }

    // 5) Total
    std::vector<double> totalAvg(steps + 1);
    for (int t = 0; t <= steps; ++t) {
        totalAvg[t] = totalSums[t] / simulations;
    }
    cashflows["Total"] = std::move(totalAvg);

    if (print == 1) 
    {
        exportCashflowsCSV(fileName, cashflows);
        std::cout
            << "Cashflows outputted succesfully" << "\n";
    }

    return cashflows;
}

