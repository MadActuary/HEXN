#pragma once

#include "Payoff.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <random>
#include <stdexcept>
#include <iomanip>

// Use an enum for duration types instead of strings
enum class DurationType { Age = 0, Visit = 1, State = 2 };

struct OutTransition {
    int toIndex;                       // target state index
    DurationType durationType;         // which duration to use
    std::vector<double> probabilities; // probability vector by duration
};

class Model {
public:
    Model(const std::string& csvFile, const std::string& initialState)
        : age(0), durationInState(0), durationSinceB(0), gen(std::random_device{}())
    {
        loadCSV(csvFile);
        // map initial state string to index
        auto it = stateIndex.find(initialState);
        if (it == stateIndex.end())
            throw std::runtime_error("Unknown initial state: " + initialState);
        currentState = it->second;
    }

    // Accessors by index
    std::string getCurrentState() const { return indexToState[currentState]; }
    size_t getAge() const { return age; }
    size_t getDurationInState() const { return durationInState; }
    size_t getDurationSinceB() const { return durationSinceB; }

    void reset(const std::string& state, size_t a, size_t dState, size_t dB) {
        auto it = stateIndex.find(state);
        if (it == stateIndex.end())
            throw std::runtime_error("Unknown reset state: " + state);
        currentState = it->second;
        age = a;
        durationInState = dState;
        durationSinceB = dB;
    }

    // Step using integer-based transitions
    void step(double uniformSample) {
        const auto& outs = transitions[currentState];
        double cumulative = 0.0;
        for (const auto& tr : outs) {
            size_t d = getDuration(tr.durationType);
            double p = (d < tr.probabilities.size() ? tr.probabilities[d] : 0.0);
            cumulative += p;
            if (uniformSample <= cumulative) {
                updateDurations(tr.toIndex);
                currentState = tr.toIndex;
                return;
            }
        }
        // if we reach here, maybe due to rounding: pick last
        if (!outs.empty()) {
            int last = outs.back().toIndex;
            updateDurations(last);
            currentState = last;
        }
    }

private:
    // internal state as index
    int currentState;
    size_t age;
    size_t durationInState;
    size_t durationSinceB;

    // mapping from state string <-> index
    std::unordered_map<std::string, int> stateIndex;
    std::vector<std::string> indexToState;

    // transitions[fromIndex] = list of outgoing transitions
    std::vector<std::vector<OutTransition>> transitions;

    // RNG
    std::mt19937 gen;

    void loadCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + filename);

        std::string line;
        // read header lines
        std::getline(file, line);
        auto fromStates = splitCSVLine(line);
        std::getline(file, line);
        auto toStates = splitCSVLine(line);
        std::getline(file, line);
        auto durations = splitCSVLine(line);

        if (fromStates.size() != toStates.size() || fromStates.size() != durations.size())
            throw std::runtime_error("CSV header misaligned");

        size_t cols = fromStates.size();
        // assign indices to each unique state
        for (size_t i = 0; i < cols; ++i) {
            const std::string& f = fromStates[i];
            const std::string& t = toStates[i];
            if (!stateIndex.count(f)) {
                stateIndex[f] = indexToState.size();
                indexToState.push_back(f);
            }
            if (!stateIndex.count(t)) {
                stateIndex[t] = indexToState.size();
                indexToState.push_back(t);
            }
        }
        int N = indexToState.size();
        transitions.assign(N, {});

        // temporary storage of columns per transition key
        struct Temp { int from, to; DurationType dt; std::vector<double> probs; };
        std::vector<Temp> tempTrans(cols);

        // initialize Temp entries
        for (size_t i = 0; i < cols; ++i) {
            tempTrans[i].from = stateIndex[fromStates[i]];
            tempTrans[i].to = stateIndex[toStates[i]];
            const auto& sdt = durations[i];
            if (sdt == "age")      tempTrans[i].dt = DurationType::Age;
            else if (sdt == "visit") tempTrans[i].dt = DurationType::Visit;
            else if (sdt == "state") tempTrans[i].dt = DurationType::State;
            else throw std::runtime_error("Unknown duration type: " + sdt);
        }

        // read rows and fill probabilities
        while (std::getline(file, line)) {
            auto vals = splitCSVLine(line);
            if (vals.size() != cols)
                throw std::runtime_error("Data row mismatch");
            for (size_t i = 0; i < cols; ++i) {
                tempTrans[i].probs.push_back(std::stod(vals[i]));
            }
        }

        // build transitions vector
        for (auto& tt : tempTrans) {
            OutTransition ot;
            ot.toIndex = tt.to;
            ot.durationType = tt.dt;
            ot.probabilities = std::move(tt.probs);
            transitions[tt.from].push_back(std::move(ot));
        }
    }

    std::vector<std::string> splitCSVLine(const std::string& line) const {
        std::istringstream ss(line);
        std::vector<std::string> toks;
        std::string tok;
        while (std::getline(ss, tok, ';')) toks.push_back(tok);
        return toks;
    }

    size_t getDuration(DurationType dt) const {
        switch (dt) {
        case DurationType::Age:   return age;
        case DurationType::Visit: return durationSinceB;
        case DurationType::State: return durationInState;
        }
        return 0;
    }

    void updateDurations(int nextIndex) {
        ++age;
        if (nextIndex == currentState) ++durationInState;
        else durationInState = 0;

        if (currentState == stateIndex["B"] || durationSinceB > 0) ++durationSinceB;
        if (nextIndex == stateIndex["B"] && nextIndex != currentState) durationSinceB = 0;
    }
};
