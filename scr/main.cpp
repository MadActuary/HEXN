#include <random>


#include "Model.h"
#include "Payoff.h"
#include "DanishPublicBenefits.h"
#include "Engine.h"


using namespace std::chrono;
sys_days today = floor<days>(system_clock::now());

int main() {


    Model model("./docs/transitions.csv", "A");

    DanishPublicBenefits mydanishPublicBenefits(100.00, 3, today);

    Engine engine(model, mydanishPublicBenefits, 10000);

    auto result = engine.getCashflow(2); // Second moment = variance component

    for (const auto& [state, values] : result) {
        std::cout << "State: " << state << "\n";
        for (int t = 0; t <= 20; ++t) {
            std::cout << "  Step " << t << ": " << values[t] << "\n";
        }
    }

    return 0;
}


