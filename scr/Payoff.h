#pragma once
#include <string>
#include <chrono>

class Payoff {
public:
    using Date = std::chrono::sys_days;
    virtual ~Payoff() = default;
    virtual double evaluate(const std::string& state, size_t duration) const = 0;
};
