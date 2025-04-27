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
#include <utility>
#include <cstddef>
#include <array>
#include <span>
#include <algorithm>

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
            return hash<std::string>()(k.from) ^ hash<std::string>()(k.to) ^ hash<std::string>()(k.durationType);
        }
    };
}

class Model {

public:

    static constexpr size_t MAX_INTENSITIES = 16;

    Model(const std::string& csvFile, const std::string& initialState)
        : currentState(initialState), age(0), durationInState(0), durationSinceB(0)
    {
        loadCSV(csvFile);
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

    std::span<const std::pair<std::string, double>> getIntensities() const {

        size_t writePos = 0;
        auto it = transitionsFromState.find(currentState);
        if (it != transitionsFromState.end()) {
            for (auto const& key : it->second) {
                if (writePos >= MAX_INTENSITIES) break;  // guard overflow
                auto pit = transitionProbabilities.find(key);
                if (pit == transitionProbabilities.end()) continue;
                auto const& values = pit->second;
                size_t d = getDuration(key.durationType);
                if (d < values.size())
                    buffer[writePos++] = { key.to, values[d] };
            }
        }
        bufferSize = writePos;
        return { buffer.data(), bufferSize };
    }

    void step(double uniformSample) {

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
    }

private:

    std::string currentState;
    size_t age;
    size_t durationInState;
    size_t durationSinceB;
    std::unordered_map<TransitionKey, std::vector<double>> transitionProbabilities;
    std::unordered_map<std::string, std::vector<TransitionKey>> transitionsFromState;

    // For getIntensities()

    mutable std::array<std::pair<std::string, double>, MAX_INTENSITIES> buffer{};
    mutable size_t bufferSize = 0;

    void loadCSV(const std::string& filename) {
        
        std::ifstream file(filename);

        if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);

        std::string line;
        std::vector<std::string> fromStates, toStates, durations;

        // Read metadata rows
        std::getline(file, line);
        fromStates = splitCSVLine(line);
        std::getline(file, line);
        toStates = splitCSVLine(line);
        std::getline(file, line);
        durations = splitCSVLine(line);

        if (fromStates.size() != toStates.size() || fromStates.size() != durations.size()) {
            throw std::runtime_error("CSV header lines are misaligned.");
        }

        // Read time-step rows
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

        for (auto const& [state, keys] : transitionsFromState) {
            if (keys.size() > bufferSize)
                bufferSize = keys.size();
            
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
    }
};

