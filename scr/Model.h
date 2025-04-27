#pragma once

#include "Payoff.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <random>
#include <stdexcept>
#include <iomanip>

struct TransitionKey {
    std::string from;
    std::string to;
    std::string durationType;

    bool operator==(const TransitionKey& other) const {
        return from == other.from && to == other.to && durationType == other.durationType;
    }
};

// Hash function for TransitionKey
namespace std {
    template <>
    struct hash<TransitionKey> {
        std::size_t operator()(const TransitionKey& k) const {
            return hash<std::string>()(k.from)
                ^ (hash<std::string>()(k.to) << 1)
                ^ (hash<std::string>()(k.durationType) << 2);
        }
    };
}

class Model {
public:
    Model(const std::string& csvFile, const std::string& initialState)
        : currentState(initialState), age(0), durationInState(0), durationSinceB(0)
    {
        loadCSV(csvFile);
        detectAbsorbingStates();
    }

    std::string getCurrentState() const { return currentState; }
    size_t getAge() const { return age; }
    size_t getDurationInState() const { return durationInState; }
    size_t getDurationSinceB() const { return durationSinceB; }

    void reset(const std::string& state, size_t a, size_t dState, size_t dB) {
        currentState = state;
        age = a;
        durationInState = dState;
        durationSinceB = dB;
    }

    const std::unordered_map<TransitionKey, std::vector<double>>& getTransitionMap() const {
        return transitionProbabilities;
    }

    std::vector<std::string> getConnectedStates() const {
        std::vector<std::string> result;
        if (isAbsorbing(currentState)) return result;
        auto it = transitionsFromState.find(currentState);
        if (it == transitionsFromState.end()) return result;

        for (const auto& key : it->second) {
            const auto& values = transitionProbabilities.at(key);
            size_t d = getDuration(key.durationType);
            if (d < values.size() && values[d] > 0.0) {
                result.push_back(key.to);
            }
        }
        return result;
    }

    std::vector<std::pair<std::string, double>>& getIntensities() const {
        intensityBuffer.clear();
        if (isAbsorbing(currentState)) return intensityBuffer;
        auto it = transitionsFromState.find(currentState);
        if (it == transitionsFromState.end()) return intensityBuffer;

        for (const auto& key : it->second) {
            const auto& values = transitionProbabilities.at(key);
            size_t d = getDuration(key.durationType);
            if (d < values.size()) {
                intensityBuffer.emplace_back(key.to, values[d]);
            }
        }
        return intensityBuffer;
    }

    void step(double uniformSample) {
        if (isAbsorbing(currentState)) {
            // Still advance durations even if state is absorbing
            updateDurations(currentState);
            return;
        }

        const auto& options = getIntensities();
        double cumulative = 0.0;
        for (const auto& [state, prob] : options) {
            cumulative += prob;
            if (uniformSample <= cumulative) {
                updateDurations(state);
                currentState = state;
                return;
            }
        }
        // If no transition taken, still update duration in same state
        updateDurations(currentState);
    }

    std::unordered_map<std::string, std::vector<double>> getTransitionProbabilities(int n) {
        const int steps = 120;
        std::unordered_map<std::string, std::vector<int>> stateCounts;
        for (const auto& [key, _] : transitionProbabilities) {
            stateCounts[key.to] = std::vector<int>(steps, 0);
            stateCounts[key.from] = std::vector<int>(steps, 0);
        }

        std::string originalState = currentState;
        size_t originalAge = age;
        size_t originalDurationInState = durationInState;
        size_t originalDurationSinceB = durationSinceB;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int sim = 0; sim < n; ++sim) {
            reset(originalState, originalAge, originalDurationInState, originalDurationSinceB);
            for (int t = 0; t < steps; ++t) {
                double sample = dis(gen);
                step(sample);
                stateCounts[currentState][t]++;
            }
        }

        std::unordered_map<std::string, std::vector<double>> result;
        for (const auto& [state, counts] : stateCounts) {
            std::vector<double> probs(steps);
            for (int t = 0; t < steps; ++t) {
                probs[t] = static_cast<double>(counts[t]) / n;
            }
            result[state] = probs;
        }
        return result;
    }

private:
    std::string currentState;
    size_t age;
    size_t durationInState;
    size_t durationSinceB;
    std::unordered_set<std::string> absorbingStates;
    std::unordered_map<TransitionKey, std::vector<double>> transitionProbabilities;
    std::unordered_map<std::string, std::vector<TransitionKey>> transitionsFromState;
    mutable std::vector<std::pair<std::string, double>> intensityBuffer;

    void loadCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
        std::string line;
        std::vector<std::string> fromStates, toStates, durations;

        std::getline(file, line);
        fromStates = splitCSVLine(line);
        std::getline(file, line);
        toStates = splitCSVLine(line);
        std::getline(file, line);
        durations = splitCSVLine(line);

        if (fromStates.size() != toStates.size() || fromStates.size() != durations.size()) {
            throw std::runtime_error("CSV header lines are misaligned.");
        }

        while (std::getline(file, line)) {
            auto values = splitCSVLine(line);
            if (values.size() != fromStates.size()) {
                throw std::runtime_error("Data row does not match header size.");
            }
            for (size_t i = 0; i < values.size(); ++i) {
                TransitionKey key{ fromStates[i], toStates[i], durations[i] };
                transitionProbabilities[key].push_back(std::stod(values[i]));
            }
        }
        for (const auto& [key, _] : transitionProbabilities) {
            transitionsFromState[key.from].push_back(key);
        }
    }

    std::vector<std::string> splitCSVLine(const std::string& line) const {
        std::istringstream stream(line);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(stream, token, ';')) {
            tokens.push_back(token);
        }
        return tokens;
    }

    size_t getDuration(const std::string& type) const {
        if (type == "age") return age;
        if (type == "visit") return durationSinceB;
        if (type == "state") return durationInState;
        throw std::runtime_error("Unknown duration type: " + type);
    }

    void updateDurations(const std::string& nextState) {
        ++age;
        if (nextState == currentState) {
            ++durationInState;
        }
        else {
            durationInState = 0;
        }
        if (currentState == "B" || durationSinceB > 0) {
            ++durationSinceB;
        }
        if (nextState == "B" && currentState != "B") {
            durationSinceB = 0;
        }
        // currentState remains unchanged here; caller updates it if needed
    }

    void detectAbsorbingStates() {
        std::unordered_set<std::string> allStates;
        for (auto& [key, _] : transitionProbabilities) {
            allStates.insert(key.from);
            allStates.insert(key.to);
        }
        for (const auto& s : allStates) {
            if (transitionsFromState.find(s) == transitionsFromState.end()) {
                absorbingStates.insert(s);
            }
        }
    }

    bool isAbsorbing(const std::string& s) const {
        return absorbingStates.count(s) > 0;
    }
};
