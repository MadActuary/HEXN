#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>

// Simple POD to hold results
struct Stats {
    double mean;      // average time in seconds
    double variance;  // population variance in seconds^2
};

// measure() takes a void() function (or lambda) and number of runs:
Stats MeasureExecution(std::function<void()> execute, int runs = 100) {
    std::vector<double> times;
    times.reserve(runs);

    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        execute();  
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        times.push_back(dt.count());
    }

    // compute mean
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / runs;

    // compute (population) variance
    double sq_sum = std::accumulate(
        times.begin(), times.end(), 0.0,
        [mean](double acc, double t) {
            double d = t - mean;
            return acc + d * d;
        }
    );
    double variance = sq_sum / runs;

    std::cout
        << "Mean:     " << mean << " s\n"
        << "Variance: " << variance << " s^2\n";

    return Stats{ mean, variance };
}