#pragma once

#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <algorithm> 

static std::string formatDouble(double value) {
    std::ostringstream oss;
    // Use fixed notation with 6 decimal places (adjust precision as needed)
    oss << std::fixed << std::setprecision(6) << value;
    std::string s = oss.str();
    // Replace decimal point with comma
    std::replace(s.begin(), s.end(), '.', ',');
    return s;
}