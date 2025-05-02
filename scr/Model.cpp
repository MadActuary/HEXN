// Model.cpp
#include "Model.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// OpenCL kernel source (embedded)
// ─────────────────────────────────────────────────────────────────────────────
static const char* stepBatchLUT_cl = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define MAX_DUR_TYPES 4

__kernel void stepBatchLUT(
    __global const double* uniforms,
    __global ushort*       curState,
    __global uint*         age,
    __global uint*         durInState,
    __global uint*         durSinceB,
    __global const ushort* lut,
    __global const uchar*  stateDType,
    const ushort           B_id,
    const uint             lutBuckets)
{
    size_t i = get_global_id(0);
    double u = uniforms[i];
    int idx = (int)(u * lutBuckets);
    if (idx < 0) idx = 0;
    else if (idx >= lutBuckets) idx = lutBuckets - 1;

    ushort s  = curState[i];
    uchar  dt = stateDType[s];
    uint   base = ((uint)s * MAX_DUR_TYPES + (uint)dt) * lutBuckets;
    ushort ns = lut[base + idx];

    age[i] += 1;
    durInState[i] = (ns == s ? durInState[i] + 1 : 0);
    if (s == B_id || durSinceB[i] > 0) durSinceB[i] += 1;
    if (ns == B_id && s != B_id)       durSinceB[i] = 0;

    curState[i] = ns;
}
)CLC";

// ─────────────────────────────────────────────────────────────────────────────
// Constructor & CSV loading
// ─────────────────────────────────────────────────────────────────────────────
Model::Model(const std::string& csvFile) {
    loadCSV(csvFile);
}

void Model::loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open " + filename);

    std::string line;
    std::vector<std::string> fromStates, toStates, durTypes;
    auto readHeader = [&](auto& out) {
        std::getline(file, line);
        std::istringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ';'))
            out.push_back(tok);
        };
    readHeader(fromStates);
    readHeader(toStates);
    readHeader(durTypes);

    size_t N = fromStates.size();
    if (toStates.size() != N || durTypes.size() != N)
        throw std::runtime_error("CSV header misaligned");

    // Read data columns
    std::vector<std::vector<double>> cols(N);
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        for (size_t i = 0; i < N; ++i) {
            std::string val;
            if (!std::getline(ss, val, ';'))
                throw std::runtime_error("Data row mismatch");
            cols[i].push_back(std::stod(val));
        }
    }

    // Build transitions & flat probability buffer
    transitions.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        StateID f = getStateID(fromStates[i]);
        StateID t = getStateID(toStates[i]);
        uint8_t d = decodeDurType(durTypes[i]);
        uint32_t len = static_cast<uint32_t>(cols[i].size());
        uint32_t off = static_cast<uint32_t>(all_probs.size());
        all_probs.insert(all_probs.end(), cols[i].begin(), cols[i].end());
        transitions.push_back({ f, t, d, off, len });
    }

    // Sort & index by 'from'
    std::sort(transitions.begin(), transitions.end(),
        [](auto const& a, auto const& b) { return a.from < b.from; });
    size_t S = stateNames.size();
    state_begin.assign(S, 0);
    state_end.assign(S, 0);
    for (size_t i = 0, T = transitions.size(); i < T; ++i) {
        StateID s = transitions[i].from;
        if (i == 0 || transitions[i - 1].from != s) {
            state_begin[s] = i;
            if (i > 0) state_end[transitions[i - 1].from] = i;
        }
        if (i == T - 1) state_end[s] = T;
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

uint8_t Model::decodeDurType(const std::string& s) const {
    if (s == "age")   return 0;
    if (s == "state") return 1;
    if (s == "visit") return 2;
    throw std::runtime_error("Unknown duration type: " + s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar stepBatch (original sampler)
// ─────────────────────────────────────────────────────────────────────────────
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
                durInState[i] = (tr.to == s ? durInState[i] + 1 : 0);
                if (s == B_id || durSinceB[i] > 0) ++durSinceB[i];
                if (tr.to == B_id && s != B_id) durSinceB[i] = 0;
                curState[i] = tr.to;
                break;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// buildLUT (CPU-only)
// ─────────────────────────────────────────────────────────────────────────────
void Model::buildLUT(int buckets) {
    lutBuckets = buckets;
    size_t S = stateNames.size();
    lut.assign(S * maxDurTypes * lutBuckets, StateID(-1));
    stateDType.assign(S, 0);

    // Record each state's duration-type
    for (size_t s = 0; s < S; ++s) {
        size_t b = state_begin[s], e = state_end[s];
        if (b < e) stateDType[s] = transitions[b].dtype;
    }

    // Build LUT per (state, dtype)
    for (size_t s = 0; s < S; ++s) {
        uint8_t dt = stateDType[s];
        size_t b = state_begin[s], e = state_end[s], K = e - b;
        if (K == 0) continue;

        // Gather probs at duration index = 0
        std::vector<double>  p(K), cdf(K);
        std::vector<StateID> tos(K);
        double total = 0.0;
        for (size_t k = 0; k < K; ++k) {
            auto const& tr = transitions[b + k];
            double pr = (0 < tr.length ? all_probs[tr.offset + 0] : 0.0);
            p[k] = pr; total += pr;
            tos[k] = tr.to;
        }
        // Normalize & build CDF
        double run = 0.0;
        for (size_t k = 0; k < K; ++k) {
            run += p[k] / (total > 0 ? total : 1.0);
            cdf[k] = run;
        }

        size_t base = (s * maxDurTypes + dt) * lutBuckets;
        for (int u = 0; u < lutBuckets; ++u) {
            double ru = double(u + 1) / double(lutBuckets);
            StateID choice = tos.back();
            for (size_t k = 0; k < K; ++k) {
                if (ru <= cdf[k]) { choice = tos[k]; break; }
            }
            lut[base + u] = choice;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// initializeBatch
// ─────────────────────────────────────────────────────────────────────────────
void Model::initializeBatch(size_t batchSize,
    const std::string& initState,
    uint32_t initAge,
    uint32_t initDurState,
    uint32_t initDurSinceB)
{
    M = batchSize;
    StateID sid = getStateID(initState);
    curState.assign(M, sid);
    age.assign(M, initAge);
    durInState.assign(M, initDurState);
    durSinceB.assign(M, initDurSinceB);
}

// ─────────────────────────────────────────────────────────────────────────────
// initOpenCL (C API) with GPU selection & logging
// ─────────────────────────────────────────────────────────────────────────────
void Model::initOpenCL() {
    if (clReady) return;
    cl_int err;

    // 1) Platform
    err = clGetPlatformIDs(1, &clPlatform, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("clGetPlatformIDs failed");

    // 2) Try GPU, else CPU
    cl_uint gpuCount = 0;
    err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
    if (err != CL_SUCCESS || gpuCount == 0) {
        std::cerr << "[OpenCL] No GPU found, falling back to CPU.\n";
        err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_CPU, 1, &clDevice, nullptr);
        if (err != CL_SUCCESS)
            throw std::runtime_error("No CPU device found");
    }
    else {
        std::vector<cl_device_id> gpus(gpuCount);
        err = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, gpuCount, gpus.data(), nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to get GPU device IDs");
        clDevice = gpus[0];
    }

    // 3) Log device name & type
    {
        char name[256];
        cl_device_type type;
        clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        clGetDeviceInfo(clDevice, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
        std::cerr << "[OpenCL] Using device: " << name
            << " (" << ((type & CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU") << ")\n";
    }

    // 4) Context
    clCtx = clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateContext failed");

    // 5) Command queue (modern API)
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    clQueue = clCreateCommandQueueWithProperties(clCtx, clDevice, props, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateCommandQueueWithProperties failed");

    // 6) Program & kernel
    const char* src = stepBatchLUT_cl;
    size_t len = std::strlen(stepBatchLUT_cl);
    clProgram = clCreateProgramWithSource(clCtx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateProgramWithSource failed");
    err = clBuildProgram(clProgram, 1, &clDevice, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        throw std::runtime_error("clBuildProgram failed:\n" + std::string(log.data()));
    }
    clKernel = clCreateKernel(clProgram, "stepBatchLUT", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateKernel failed");

    // 7) Buffers & args
    buf_uniforms = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, sizeof(double) * M, nullptr, &err);
    buf_curState = clCreateBuffer(clCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(StateID) * M, curState.data(), &err);
    buf_age = clCreateBuffer(clCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * M, age.data(), &err);
    buf_durInState = clCreateBuffer(clCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * M, durInState.data(), &err);
    buf_durSinceB = clCreateBuffer(clCtx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(uint32_t) * M, durSinceB.data(), &err);
    buf_lut = clCreateBuffer(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(StateID) * lut.size(), lut.data(), &err);
    buf_stateDType = clCreateBuffer(clCtx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(uint8_t) * stateDType.size(),
        stateDType.data(), &err);
    if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer failed");

    int arg = 0;
    clSetKernelArg(clKernel, arg++, sizeof(buf_uniforms), &buf_uniforms);
    clSetKernelArg(clKernel, arg++, sizeof(buf_curState), &buf_curState);
    clSetKernelArg(clKernel, arg++, sizeof(buf_age), &buf_age);
    clSetKernelArg(clKernel, arg++, sizeof(buf_durInState), &buf_durInState);
    clSetKernelArg(clKernel, arg++, sizeof(buf_durSinceB), &buf_durSinceB);
    clSetKernelArg(clKernel, arg++, sizeof(buf_lut), &buf_lut);
    clSetKernelArg(clKernel, arg++, sizeof(buf_stateDType), &buf_stateDType);
    StateID B_id = stateIndex.count("B") ? stateIndex.at("B") : StateID(-1);
    clSetKernelArg(clKernel, arg++, sizeof(B_id), &B_id);
    clSetKernelArg(clKernel, arg++, sizeof(uint32_t), &lutBuckets);

    clReady = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// stepBatchLUT → dispatch to OpenCL
// ─────────────────────────────────────────────────────────────────────────────
void Model::stepBatchLUT(const double* uniforms) {
    if (!clReady) initOpenCL();
    cl_int err;

    // Write uniforms
    err = clEnqueueWriteBuffer(clQueue, buf_uniforms, CL_FALSE,
        0, sizeof(double) * M, uniforms,
        0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueWriteBuffer failed");

    // Launch kernel
    size_t global = M;
    err = clEnqueueNDRangeKernel(clQueue, clKernel,
        1, nullptr,
        &global, nullptr,
        0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueNDRangeKernel failed");

    // Read back
    clEnqueueReadBuffer(clQueue, buf_curState, CL_TRUE, 0,
        sizeof(StateID) * M, curState.data(),
        0, nullptr, nullptr);
    clEnqueueReadBuffer(clQueue, buf_age, CL_TRUE, 0,
        sizeof(uint32_t) * M, age.data(),
        0, nullptr, nullptr);
    clEnqueueReadBuffer(clQueue, buf_durInState, CL_TRUE, 0,
        sizeof(uint32_t) * M, durInState.data(),
        0, nullptr, nullptr);
    clEnqueueReadBuffer(clQueue, buf_durSinceB, CL_TRUE, 0,
        sizeof(uint32_t) * M, durSinceB.data(),
        0, nullptr, nullptr);

    clFinish(clQueue);
}
