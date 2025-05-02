#include "Model.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

Model::Model(const std::string& csvFile) {
    loadCSV(csvFile);
}

void Model::loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open " + filename);

    std::string line;
    std::vector<std::string> fromStates, toStates, durTypes;
    auto readHeader = [&](std::vector<std::string>& out) {
        std::getline(file, line);
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ';')) out.push_back(tok);
        };
    readHeader(fromStates);
    readHeader(toStates);
    readHeader(durTypes);

    size_t N = fromStates.size();
    if (toStates.size() != N || durTypes.size() != N)
        throw std::runtime_error("CSV header misaligned");

    std::vector<std::vector<double>> cols(N);
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string val;
        for (size_t i = 0; i < N; ++i) {
            if (!std::getline(ss, val, ';'))
                throw std::runtime_error("Data row mismatch");
            cols[i].push_back(std::stod(val));
        }
    }

    transitions.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        StateID f = getStateID(fromStates[i]);
        StateID t = getStateID(toStates[i]);
        DurType d = decodeDurType(durTypes[i]);
        uint32_t len = static_cast<uint32_t>(cols[i].size());
        uint32_t off = static_cast<uint32_t>(all_probs.size());
        all_probs.insert(all_probs.end(), cols[i].begin(), cols[i].end());
        transitions.push_back({ f, t, d, off, len });
    }

    std::sort(transitions.begin(), transitions.end(),
        [](auto const& a, auto const& b) { return a.from < b.from; });
    size_t S = stateNames.size();
    state_begin.assign(S, 0);
    state_end.assign(S, 0);
    for (size_t i = 0, T = transitions.size(); i < T; ++i) {
        auto s = transitions[i].from;
        if (i == 0 || transitions[i - 1].from != s) {
            state_begin[s] = i;
            if (i > 0)
                state_end[transitions[i - 1].from] = i;
        }
        if (i == T - 1)
            state_end[s] = T;
    }
}

Model::StateID Model::getStateID(const std::string& s) {
    auto it = stateIndex.find(s);
    if (it != stateIndex.end()) return it->second;
    StateID id = static_cast<StateID>(stateNames.size());
    stateIndex[s] = id;
    stateNames.push_back(s);
    return id;
}

Model::DurType Model::decodeDurType(const std::string& s) const {
    if (s == "age")   return 0;
    if (s == "state") return 1;
    if (s == "visit") return 2;
    throw std::runtime_error("Unknown duration type: " + s);
}

void Model::initializeBatch(size_t batchSize,
    const std::string& initState,
    uint32_t initAge,
    uint32_t initDurState,
    uint32_t initDurSinceB) {
    M = batchSize;
    StateID sid = getStateID(initState);
    curState.assign(M, sid);
    age.assign(M, initAge);
    durInState.assign(M, initDurState);
    durSinceB.assign(M, initDurSinceB);
}

void Model::stepBatch(const double* uniforms) {
    StateID B_id = stateIndex.count("B") ? stateIndex.at("B") : StateID(-1);
    for (size_t i = 0; i < M; ++i) {
        StateID s = curState[i];
        size_t b = state_begin[s], e = state_end[s];
        uint32_t da = age[i], ds = durInState[i], dv = durSinceB[i];
        double u = uniforms[i], cum = 0.0;
        for (size_t j = b; j < e; ++j) {
            auto const& tr = transitions[j];
            uint32_t d = (tr.dtype == 0 ? da : tr.dtype == 1 ? ds : dv);
            double p = (d < tr.length ? all_probs[tr.offset + d] : 0.0);
            cum += p;
            if (u <= cum) {
                ++age[i];
                if (tr.to == s) durInState[i]++; else durInState[i] = 0;
                if (s == B_id || durSinceB[i] > 0) durSinceB[i]++;
                if (tr.to == B_id && s != B_id) durSinceB[i] = 0;
                curState[i] = tr.to;
                break;
            }
        }
    }
}