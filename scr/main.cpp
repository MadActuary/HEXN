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

    Model model("../docs/transitions.csv", "A");

    sys_days today = floor<days>(system_clock::now());

    DanishPublicBenefits mydanishPublicBenefits(100, 0, today);

    Engine engine(model, mydanishPublicBenefits, 100000);

    auto result = engine.getCashflow(1, 120, true, "../docs/expectedPayments.CSV"); // Second moment = variance component

}

int main() {

    //execute();

    //Time measurement command
    auto stats = MeasureExecution(execute, 100, "New benchmark");

    return 0;
}


