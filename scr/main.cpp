#include <random>

#include "Model.h"
#include "Payoff.h"
#include "Engine.h"

int main() {


    Model model("./docs/transitions.csv", "A");

    Payoff payoff(100);

    Engine engine(model, payoff, 10000);

    auto result = engine.getCashflow(2); // Second moment = variance component

    for (const auto& [state, values] : result) {
        std::cout << "State: " << state << "\n";
        for (int t = 0; t <= 20; ++t) {
            std::cout << "  Step " << t << ": " << values[t] << "\n";
        }
    }

    return 0;
}


