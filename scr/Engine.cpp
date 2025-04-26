#include "Engine.h"

Engine::Engine(Model& model, const Payoff& payoff, int simulations)
    : model(model), payoff(payoff), simulations(simulations) {}

std::unordered_map<std::string, std::vector<double>> Engine::getCashflow(int moment, int steps) {
    // Dynamic counts: initialize when first encountered
    std::unordered_map<std::string, std::vector<int>> stateCounts;

    // Save original model state
    std::string originalState = model.getCurrentState();
    size_t originalAge = model.getAge();
    size_t originalInState = model.getDurationInState();
    size_t originalSinceB = model.getDurationSinceB();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < simulations; ++i) {
        // Reset model
        model.reset(originalState, originalAge, originalInState, originalSinceB);
        // record initial state at t=0
        {
            auto s = model.getCurrentState();
            auto& counts = stateCounts.emplace(s, std::vector<int>(steps + 1, 0)).first->second;
            counts[0]++;
        }
        // simulate path
        for (int t = 1; t <= steps; ++t) {
            double u = dis(gen);
            model.step(u);
            auto s = model.getCurrentState();
            auto& counts = stateCounts.emplace(s, std::vector<int>(steps + 1, 0)).first->second;
            counts[t]++;
        }
    }

    // Build cashflows
    std::unordered_map<std::string, std::vector<double>> cashflows;
    std::vector<double> total(steps + 1, 0.0);

    for (auto& entry : stateCounts) {
        const auto& state = entry.first;
        const auto& counts = entry.second;
        std::vector<double> values(steps + 1, 0.0);
        for (int t = 0; t <= steps; ++t) {
            double prob = static_cast<double>(counts[t]) / simulations;
            double val = payoff.evaluate(state, model.getDurationInState());
            values[t] = prob * val;
            total[t] += values[t];
        }
        cashflows[state] = std::move(values);
    }
    cashflows["Total"] = std::move(total);

    return cashflows;
}
