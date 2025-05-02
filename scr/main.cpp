// main.cpp
#include "Model.h"
#include "Engine.h"
#include "Payoff.h"
#include "ExecutionStats.h"

#include <iostream>
#include <string>

// Example Payoff: reward = 1.0 for state "B", 0.0 otherwise
struct DemoPayoff : Payoff {
    double evaluate(const std::string& state, size_t duration) const override {
        return 100.0;
    }
};

void execution() {
    std::string csvFile = "../docs/transitions.csv";
    std::string initialState = "A";
    int         steps = 120;
    int         sims = 100000;
    std::string outFile = "../docs/test.csv";

    // 1) Load transition model
    Model model(csvFile);
    // 2) Initialize a single "prototype" path to capture origin state/durations
    model.initializeBatch(
        1, 
        initialState, 
        /*age=*/0, 
        /*durState=*/0, 
        /*durSinceB=*/0);

    // 3) Set up payoff and engine
    DemoPayoff payoff;
    Engine engine(model, payoff, sims);

    // 4) Run Monte Carlo, moment=1 (expected value)
    auto cashflows = engine.getCashflow(
        1,
        steps,
        true,
        outFile
    );

}

int main() {

    //execution();

    MeasureExecution(execution, 10, "Test");
   
}
