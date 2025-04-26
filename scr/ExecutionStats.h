#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <functional>
#include <windows.h>
#include <psapi.h>
#include <ctime>
#include <string>

#include "formatDouble.h"

// Simple POD to hold results
struct Stats {
    int runs;                 // number of runs
    double meanTime;          // average execution time in seconds
    double varianceTime;      // population variance in seconds^2
    size_t peakMemoryBytes;   // peak memory usage in bytes (OS reported)
};

// Get peak resident memory (working set) in bytes
size_t getPeakRSS() {
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.PeakWorkingSetSize;
}

// Escape a string for CSV: double any '"' and wrap the whole in quotes
static std::string csvEscape(const std::string& s) {
    std::ostringstream oss;
    oss << '"';
    for (char c : s) {
        if (c == '"') oss << "\"\"";
        else           oss << c;
    }
    oss << '"';
    return oss.str();
}

// Measure execution time, peak memory, and append a CSV row to performance_log.txt.
// Columns: Date,Comment,Runs,MeanTime,VarianceTime,PeakMemoryBytes
Stats MeasureExecution(std::function<void()> execute,
    int runs = 100,
    const std::string& comment = "")
{
    // ——— benchmarking ———
    std::vector<double> times;
    times.reserve(runs);

    for (int i = 0; i < runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        execute();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
    }

    double meanTime = std::accumulate(times.begin(), times.end(), 0.0) / runs;
    double varianceTime = std::accumulate(
        times.begin(), times.end(), 0.0,
        [meanTime](double acc, double t) {
            double d = t - meanTime;
            return acc + d * d;
        }
    ) / runs;
    size_t peakMemory = getPeakRSS();

   
    // ——— CSV logging ———
    const char* filename = "../docs/performance_log.csv";
    bool writeHeader = false;

    // check if file is new/empty
    {
        std::ifstream ifs(filename);
        if (!ifs.good() || ifs.peek() == std::ifstream::traits_type::eof()) {
            writeHeader = true;
        }
    }

    std::ofstream ofs(filename, std::ios::app);
    if (!ofs) {
        std::cerr << "Error: could not open " << filename << " for appending\n";
    }
    else {
        if (writeHeader) {
            ofs << "Date;Comment;Runs(n);MeanTime(s);VarianceTime;PeakMemory(MB)\n";
        }

        // get today's date
        std::time_t t = std::time(nullptr);
        std::tm localTm;
        localtime_s(&localTm, &t);
        char dateBuf[11]; // "YYYY-MM-DD\0"
        std::strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d", &localTm);

        // write one CSV row
        ofs
            << dateBuf << ';'
            << csvEscape(comment) << ';'
            << runs << ';'
            << formatDouble(meanTime) << ';'
            << formatDouble(varianceTime) << ';'
            << formatDouble(peakMemory / (1024.0 * 1024.0))
            << "\n";


        // ——— console output ———
        std::cout
            << "Performance logged succesfully" << "\n";

    }

    return Stats{ runs, meanTime, varianceTime, peakMemory };
}
