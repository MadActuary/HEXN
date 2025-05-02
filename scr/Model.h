#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Model {
public:
    using StateID = uint16_t;
    using DurType = uint8_t; // 0=age,1=state,2=visit

    Model(const std::string& csvFile);

    void initializeBatch(size_t batchSize,
        const std::string& initState,
        uint32_t initAge = 0,
        uint32_t initDurState = 0,
        uint32_t initDurSinceB = 0);

    void stepBatch(const double* uniforms);

    const std::vector<StateID>& getCurrentStates() const { return curState; }
    const std::vector<uint32_t>& getDurationsInState() const { return durInState; }
    const std::vector<std::string>& getStateNames() const { return stateNames; }

private:
    struct Trans {
        StateID    from, to;
        DurType    dtype;
        uint32_t   offset, length;
    };

    std::vector<double>       all_probs;
    std::vector<Trans>        transitions;
    std::vector<size_t>       state_begin, state_end;
    std::unordered_map<std::string, StateID> stateIndex;
    std::vector<std::string>  stateNames;

    size_t                  M = 0;
    std::vector<StateID>    curState;
    std::vector<uint32_t>   age, durInState, durSinceB;

    void loadCSV(const std::string& filename);
    StateID getStateID(const std::string& s);
    DurType decodeDurType(const std::string& s) const;
};
