#include "Engine.h"

Engine::Engine(Model& model, const Payoff& payoff, int simulations)
    : model(model), payoff(payoff), simulations(simulations) {}

std::unordered_map<std::string, std::vector<double>> Engine::getCashflow(int moment, int steps) {
    std::unordered_map<std::string, std::vector<int>> stateCounts;
    for (const auto& [key, _] : model.getTransitionMap()) {
        stateCounts[key.from] = std::vector<int>(steps + 1, 0);
        stateCounts[key.to] = std::vector<int>(steps + 1, 0);
    }

    std::string originalState = model.getCurrentState();
    size_t originalAge = model.getAge();
    size_t originalInState = model.getDurationInState();
    size_t originalSinceB = model.getDurationSinceB();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < simulations; ++i) {
        model.reset(originalState, originalAge, originalInState, originalSinceB);
        stateCounts[model.getCurrentState()][0]++;

        for (int t = 1; t <= steps; ++t) {
            model.step(dis(gen));
            stateCounts[model.getCurrentState()][t]++;
        }
    }

    std::unordered_map<std::string, std::vector<double>> cashflows;

    std::vector<double> total(steps + 1, 0.0);

    for (const auto& [state, counts] : stateCounts) {
        std::vector<double> values(steps + 1, 0.0);
        for (int t = 0; t <= steps; ++t) {
            double prob = static_cast<double>(counts[t]) / simulations;
            double val = payoff.evaluate(state, moment);
            values[t] = prob * val;
            total[t] += values[t];
        }
        cashflows[state] = values;
    }

    cashflows["Total"] = total;

    return cashflows;

}




