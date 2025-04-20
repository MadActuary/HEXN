#pragma once
#include <string>
#include "Payoff.h"

class DanishPublicBenefits : public Payoff {
public:
    // use the base class's Date alias so there's no chance of mismatch
    using Date = Payoff::Date;

    DanishPublicBenefits(double amount,
        int waitingMonths,
        Date referenceDate)
        : amount_{ amount }
        , waitingMonths_{ waitingMonths }
        , referenceDate_{ referenceDate }
    {}

    // NOTE: signature here exactly matches Payoff::evaluate(...)
    double evaluate(const std::string& state,
        size_t duration) const override
    {
        return (duration < waitingMonths_)
            ? 0.0
            : amount_;
    }

private:
    double amount_;
    int    waitingMonths_;
    Date   referenceDate_;
};
