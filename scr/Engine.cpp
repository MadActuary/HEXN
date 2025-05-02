#include "Engine.h"
#include "formatDouble.h"
#include <random>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <iostream>

Engine::Engine(Model& model, const Payoff& payoff, int simulations)
    : model(model), payoff(payoff), simulations(simulations) {}

std::unordered_map<std::string, std::vector<double>>
Engine::getCashflow(int moment, int steps, bool print, const std::string& fileName) {
    auto initStates = model.getCurrentStates();
    auto initDurState = model.getDurationsInState();
    std::string origState = model.getStateNames()[initStates[0]];
    uint32_t origDur = initDurState[0];
    model.initializeBatch(static_cast<size_t>(simulations), origState, 0, origDur, 0);

    std::unordered_map<std::string, std::vector<double>> sums;
    std::vector<double> totalSums(static_cast<size_t>(steps) + 1, 0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    size_t M = static_cast<size_t>(simulations);
    std::vector<double> uniforms(static_cast<size_t>(steps) * M);
    for (auto& u : uniforms) u = dis(gen);

    const auto& names = model.getStateNames();
    const auto& states0 = model.getCurrentStates();
    const auto& durs0 = model.getDurationsInState();
    for (size_t i = 0; i < M; ++i) {
        std::string s = names[states0[i]];
        double pv = payoff.evaluate(s, durs0[i]);
        auto& v = sums.emplace(s, std::vector<double>(static_cast<size_t>(steps) + 1, 0.0)).first->second;
        v[0] += pv;
        totalSums[0] += pv;
    }

    for (int t = 1; t <= steps; ++t) {
        model.stepBatch(&uniforms[static_cast<size_t>(t - 1) * M]);
        const auto& sts = model.getCurrentStates();
        const auto& durs = model.getDurationsInState();
        for (size_t i = 0; i < M; ++i) {
            std::string s = names[sts[i]];
            double pv = std::pow(payoff.evaluate(s, durs[i]), static_cast<double>(moment));
            auto& v = sums.emplace(s, std::vector<double>(static_cast<size_t>(steps) + 1, 0.0)).first->second;
            v[static_cast<size_t>(t)] += pv;
            totalSums[static_cast<size_t>(t)] += pv;
        }
    }

    std::unordered_map<std::string, std::vector<double>> cashflows;
    for (auto& kv : sums) {
        auto& sumV = kv.second;
        std::vector<double> avg(sumV.size());
        for (size_t t = 0; t < sumV.size(); ++t) {
            avg[t] = sumV[t] / static_cast<double>(simulations);
        }
        cashflows[kv.first] = std::move(avg);
    }

    std::vector<double> tot(totalSums.size());
    for (size_t t = 0; t < totalSums.size(); ++t) {
        tot[t] = totalSums[t] / static_cast<double>(simulations);
    }
    cashflows["Total"] = std::move(tot);

    if (print) {
        exportCashflowsCSV(fileName, cashflows);
        std::cout << "Cashflows outputted successfully\n";
    }
    return cashflows;
}

void Engine::exportCashflowsCSV(const std::string& filename,
    const std::unordered_map<std::string, std::vector<double>>& cashflows) {
    std::vector<std::string> states;
    for (auto const& kv : cashflows) if (kv.first != "Total") states.push_back(kv.first);
    std::sort(states.begin(), states.end());
    states.push_back("Total");

    std::ofstream ofs(filename);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open " + filename);
    for (size_t i = 0; i < states.size(); ++i) {
        ofs << states[i] << (i + 1 < states.size() ? ";" : "\n");
    }
    size_t T = cashflows.begin()->second.size();
    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < states.size(); ++i) {
            ofs << formatDouble(cashflows.at(states[i])[t])
                << (i + 1 < states.size() ? ";" : "\n");
        }
    }
}
