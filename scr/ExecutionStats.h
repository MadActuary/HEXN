#pragma once

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include <windows.h>
#include <psapi.h>

// Simple POD to hold results
struct Stats {
    int runs;         // average execution time in seconds
    double meanTime;         // average execution time in seconds
    double varianceTime;     // population variance in seconds^2
    size_t peakMemoryBytes;  // peak memory usage in bytes (OS reported)
};

// Get peak resident memory (working set) in bytes
size_t getPeakRSS() {
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.PeakWorkingSetSize;
}

// Measure execution time and peak memory usage
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

    // Compute mean
    double meanTime = std::accumulate(times.begin(), times.end(), 0.0) / runs;

    // Compute (population) variance
    double varianceTime = std::accumulate(
        times.begin(), times.end(), 0.0,
        [meanTime](double acc, double t) {
            double d = t - meanTime;
            return acc + d * d;
        }
    ) / runs;

    // Get peak memory
    size_t peakMemory = getPeakRSS();

    std::cout
        << "Runs:          " << runs << " s\n"
        << "Mean Time:     " << meanTime << " s\n"
        << "Variance Time: " << varianceTime << " s^2\n"
        << "Peak Memory:   " << (float) peakMemory / (1024*1024) << " MB\n";

    return Stats{runs, meanTime, varianceTime, peakMemory };
}
