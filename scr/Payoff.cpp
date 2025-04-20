#include "Payoff.h"

#include <cmath>

Payoff::Payoff(int payment) : paymentAmount(payment) {}

double Payoff::evaluate(const std::string& state, int moment) const {
    
    if (moment <= 0) return 0.0;
    
    if (state != "D"){
       
        return std::pow(static_cast<double>(paymentAmount), moment);
    
    } else {
        return 0.0;
    }
   
}   