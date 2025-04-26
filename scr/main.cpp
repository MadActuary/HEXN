#include "Model.h"
#include "Payoff.h"
#include "DanishPublicBenefits.h"
#include "Engine.h"
#include "ExecutionStats.h"

#include <random>
#include <numeric>
#include <cmath>  

using namespace std::chrono;

void execute() {

    bool print = 0;

    Model model("../docs/transitions.csv", "A");

    sys_days today = floor<days>(system_clock::now());

    DanishPublicBenefits mydanishPublicBenefits(100.00, 3, today);

    Engine engine(model, mydanishPublicBenefits, 100000);

    auto result = engine.getCashflow(1); // Second moment = variance component

    if (print == 1) {
        for (const auto& [state, values] : result) {
            std::cout << "State: " << state << "\n";
            for (int t = 0; t <= 120; ++t) {
                std::cout << "  Step " << t << ": " << values[t] << "\n";
            }
        }
    }
}

int main() {

    //execute();

    //Time measurement command
    auto stats = MeasureExecution(execute, 100);

    return 0;
}


