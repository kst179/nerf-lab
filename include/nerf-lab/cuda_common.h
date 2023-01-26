/**
 * @file cuda_common.h
 * @author Kolovtsev Konstantin (kozlovtsev179@gmail.com)
 * @brief Common cuda functions 
 * @version 0.1
 * @date 2023-01-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <tiny-cuda-nn/common.h>

namespace nerf {

constexpr int CUDA_NUM_BANKS = 32;
constexpr int CUDA_WARP_SIZE = 32;

constexpr int const_log2(int n) { return n < 2 ? 0 : const_log2(n / 2) + 1; }
constexpr int CUDA_LOG_NUM_BANKS = const_log2(CUDA_NUM_BANKS);

template<typename T>
__device__ inline
int bank_conflict_free_offset(int i) {
    return (i >> (CUDA_LOG_NUM_BANKS * (4 / sizeof(T))));
}

template <typename T, int BLOCK_SIZE, bool FORWARD = true>
__device__ inline
float blockwise_cumulative_sum(T thread_value) {
    static_assert(BLOCK_SIZE && !(BLOCK_SIZE & (BLOCK_SIZE - 1)), "BLOCK_SIZE should be a power of 2");
    assert(BLOCK_SIZE == blockDim.x);
    
    __shared__ T data[BLOCK_SIZE + (BLOCK_SIZE >> CUDA_LOG_NUM_BANKS)];
   
    int tid = threadIdx.x;

    // copy data to shared memory
    int i;
    if (FORWARD) {
        i = tid;
    } else {
        i = BLOCK_SIZE - tid - 1;
    }

    i += bank_conflict_free_offset<T>(i);

    data[i] = thread_value;
    __syncthreads();

    // up sweep
    #pragma unroll
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int k = (tid + 1) * (stride << 1) - 1;
        int j = k - stride;

        if (k < BLOCK_SIZE) {
            k += bank_conflict_free_offset<T>(k);
            j += bank_conflict_free_offset<T>(j);

            data[k] += data[j];
        }
        __syncthreads();
    }

    // zero last element
    if (tid == 0) {
        int k = BLOCK_SIZE - 1;
        k += bank_conflict_free_offset<T>(k);

        data[k] = 0;
    }
    __syncthreads();

    // down sweep
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride != 0; stride >>= 1) {
        int k = (tid + 1) * (stride << 1) - 1;
        int j = k - stride;

        if (k < BLOCK_SIZE) {
            k += bank_conflict_free_offset<T>(k);
            j += bank_conflict_free_offset<T>(j);

            float t = data[k];
            data[k] += data[j];
            data[j] = t;
        }
        __syncthreads();
    }

    return data[i];
}

template<typename T, int BLOCK_SIZE>
__device__ inline
float blockwise_sum(T thread_value) {
    static_assert(BLOCK_SIZE && !(BLOCK_SIZE & (BLOCK_SIZE - 1)), "BLOCK_SIZE should be a power of 2");
    assert(BLOCK_SIZE == blockDim.x);

    __shared__ volatile T sh_data[BLOCK_SIZE];

    int tid = threadIdx.x;

    sh_data[tid] = thread_value;
    __syncthreads();

    if (BLOCK_SIZE > 2 * CUDA_WARP_SIZE) {
        #pragma unroll
        for (int offset = BLOCK_SIZE / 2; offset > CUDA_WARP_SIZE; offset >>= 1) {
            if (tid < offset) {
                sh_data[tid] += sh_data[tid + offset];
            }
            __syncthreads();
        }
    }

    // here goes single warp of synced threads
    // so we don't do any syncing, neither branching
    // printf("BLOCK_SIZE %d, WARP_SIZE %d\n", BLOCK_SIZE, WARP_SIZE);
    if (tid < min(BLOCK_SIZE / 2, CUDA_WARP_SIZE)) {
        #pragma unroll
        for (int offset = min(BLOCK_SIZE / 2, CUDA_WARP_SIZE); offset > 0; offset >>= 1) {
            // printf("tid: %d, offset: %d, sh[tid]: %f, sh[tid+offset]: %f\n", tid, offset, sh_data[tid], sh_data[tid+offset]);
            sh_data[tid] += sh_data[tid + offset];
            // printf("tid: %d, sh[tid]: %f\n", tid, sh_data[tid]);
        }
    }

    // printf("tid: %d\n", tid);

    // sync with other threads outside the last warp
    __syncthreads();

    return sh_data[0];
}

}