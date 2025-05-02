// Model.h
#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <CL/cl.h>

class Model {
public:
    using StateID = uint16_t;

    // Load CSV transition data
    Model(const std::string& csvFile);

    // Build the LUT on CPU (one-time)
    void buildLUT(int buckets = 1024);

    // Initialize batch of M individuals (must be called before stepping)
    void initializeBatch(size_t batchSize,
        const std::string& initState,
        uint32_t initAge = 0,
        uint32_t initDurState = 0,
        uint32_t initDurSinceB = 0);

    // Original scalar step (fallback)
    void stepBatch(const double* uniforms);

    // Fast LUT-based step, executed on the first available OpenCL device
    void stepBatchLUT(const double* uniforms);

    // Accessors
    const std::vector<StateID>& getCurrentStates()    const { return curState; }
    const std::vector<uint32_t>& getDurationsInState() const { return durInState; }
    const std::vector<std::string>& getStateNames()   const { return stateNames; }

private:
    static constexpr int maxDurTypes = 4;

    struct Trans {
        StateID from, to;
        uint8_t dtype;      // 0=age,1=state,2=visit,...
        uint32_t offset, length;
    };

    // CSV-loaded data
    std::vector<double>            all_probs;
    std::vector<Trans>             transitions;
    std::vector<size_t>            state_begin, state_end;
    std::unordered_map<std::string, StateID> stateIndex;
    std::vector<std::string>       stateNames;
    std::vector<uint8_t>           stateDType;   // per-state duration-type

    // LUT data
    int                             lutBuckets = 0;
    std::vector<StateID>            lut;          // size = S * maxDurTypes * lutBuckets

    // Batch of individuals
    size_t                          M = 0;
    std::vector<StateID>            curState;
    std::vector<uint32_t>           age, durInState, durSinceB;

    // OpenCL C-API objects
    bool                            clReady = false;
    cl_platform_id                  clPlatform;
    cl_device_id                    clDevice;
    cl_context                      clCtx;
    cl_command_queue                clQueue;
    cl_program                      clProgram;
    cl_kernel                       clKernel;
    cl_mem                          buf_uniforms,
        buf_curState,
        buf_age,
        buf_durInState,
        buf_durSinceB,
        buf_lut,
        buf_stateDType;

    // Helpers
    void loadCSV(const std::string& filename);
    StateID getStateID(const std::string& s);
    uint8_t decodeDurType(const std::string& s) const;
    void initOpenCL();
};
