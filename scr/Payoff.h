#pragma once

#include <string>


class Payoff
{

public:

	explicit Payoff(int payment);

	double evaluate(const std::string& state, int moment = 1) const;

private: 

	int paymentAmount;

};

