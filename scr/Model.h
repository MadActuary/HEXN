// Model.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Model {
public:
    using StateID = uint16_t;

    // Load CSV transition data
    Model(const std::string& csvFile);

    // Initialize m independent individuals
    void initializeBatch(size_t batchSize,
        const std::string& initState,
        uint32_t initAge = 0,
        uint32_t initDurState = 0,
        uint32_t initDurSinceB = 0);

    // Original scalar step
    void stepBatch(const double* uniforms);

    // Build the LUT (one‐time; e.g. m.buildLUT(1024))
    void buildLUT(int buckets = 2048);

    // Fast LUT-based step
    void stepBatchLUT(const double* uniforms);

    // Accessors
    const std::vector<StateID>& getCurrentStates() const { return curState; }
    const std::vector<uint32_t>& getDurationsInState() const { return durInState; }
    const std::vector<std::string>& getStateNames()   const { return stateNames; }

private:
    static constexpr int maxDurTypes = 4;  // up to 4 duration‐types

    struct Trans {
        StateID from, to;
        uint8_t dtype;      // 0=age,1=state,2=visit,...
        uint32_t offset, length;
    };

    // CSV‐loaded data
    std::vector<double>       all_probs;
    std::vector<Trans>        transitions;
    std::vector<size_t>       state_begin, state_end;
    std::unordered_map<std::string, StateID> stateIndex;
    std::vector<std::string>  stateNames;

    // Per‐state duration‐type (all outgoing Trans from the same state share dtype)
    std::vector<uint8_t>      stateDType;

    // Batch of individuals
    size_t                    M = 0;
    std::vector<StateID>      curState;
    std::vector<uint32_t>     age, durInState, durSinceB;

    // LUT data: [ state * maxDurTypes + dtype ] * buckets + bucketIdx
    int                       lutBuckets = 0;
    std::vector<StateID>      lut;

    // CSV loading helpers
    void loadCSV(const std::string& filename);
    StateID getStateID(const std::string& s);
    uint8_t decodeDurType(const std::string& s) const;
};
