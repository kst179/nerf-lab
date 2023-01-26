#include <gtest/gtest.h>
#include <random>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <nerf-lab/cuda_common.h>

constexpr float EPS = 1e-4;

template<int BLOCK_SIZE>
__global__ void blockwise_cumsum_kernel(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = nerf::blockwise_cumulative_sum<float, BLOCK_SIZE>(data[i]);
}

template<int BLOCK_SIZE>
__global__ void blockwise_sum_kernel(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = nerf::blockwise_sum<float, BLOCK_SIZE>(data[i]);
}

template<int N, int BLOCK_SIZE>
void test_blockwise_cumsum() {
    std::vector<float> data(N);
    std::vector<float> expected(N);
    std::vector<float> actual(N);

    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0, 1);

    for (int i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    for (int j = 0; j < N; j += BLOCK_SIZE) {
        expected[j] = 0;
    }

    for (int i = 1; i < BLOCK_SIZE; ++i) {
        for (int j = i; j < N; j += BLOCK_SIZE) {
            expected[j] = expected[j - 1] + data[j - 1];
        }
    }

    tcnn::GPUMemory<float> data_gpu(N);
    data_gpu.copy_from_host(data);

    blockwise_cumsum_kernel<BLOCK_SIZE><<<N / BLOCK_SIZE, BLOCK_SIZE>>>(data_gpu.data());
    cudaDeviceSynchronize();

    data_gpu.copy_to_host(actual);

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(expected[i], actual[i], EPS) << "expected and actual arrays differ at index " << i;
    }
}

template<int N, int BLOCK_SIZE>
void test_blockwise_sum() {
    std::vector<float> data(N);
    std::vector<float> expected(N);
    std::vector<float> actual(N);

    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0, 1);

    for (int i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    for (int i = 0; i < N; i += BLOCK_SIZE) {
        float sum = 0;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += data[i + j];
        }

        for (int j = 0; j < BLOCK_SIZE; ++j) {
            expected[i + j] = sum;
        }
    }

    tcnn::GPUMemory<float> data_gpu(N);
    data_gpu.copy_from_host(data);

    blockwise_sum_kernel<BLOCK_SIZE><<<N / BLOCK_SIZE, BLOCK_SIZE>>>(data_gpu.data());
    cudaDeviceSynchronize();

    data_gpu.copy_to_host(actual);

    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(expected[i], actual[i], EPS) << "expected and actual arrays differ at index " << i;
    }
}

TEST(BlockwiseCumSum, SingleBlockTiny)      { test_blockwise_cumsum<1, 1>(); }
TEST(BlockwiseCumSum, SingleBlockSmall)     { test_blockwise_cumsum<4, 4>(); }
TEST(BlockwiseCumSum, SingleBlockMedium)    { test_blockwise_cumsum<32, 32>(); }
TEST(BlockwiseCumSum, SingleBlockLarge)     { test_blockwise_cumsum<512, 512>(); }
TEST(BlockwiseCumSum, SeveralBlocksTiny)    { test_blockwise_cumsum<1024, 1>(); }
TEST(BlockwiseCumSum, SeveralBlocksMedium)  { test_blockwise_cumsum<1024, 32>(); }
TEST(BlockwiseCumSum, SeveralBlocksLarge)   { test_blockwise_cumsum<1024, 512>(); }
TEST(BlockwiseCumSum, ManyBlocksLarge)      { test_blockwise_cumsum<(1 << 20), 512>(); }

TEST(BlockwiseSum, SingleBlockTiny)         { test_blockwise_sum<1, 1>(); }
TEST(BlockwiseSum, SingleBlockSmall)        { test_blockwise_sum<4, 4>(); }
TEST(BlockwiseSum, SingleBlockMedium)       { test_blockwise_sum<32, 32>(); }
TEST(BlockwiseSum, SingleBlockLarge)        { test_blockwise_sum<512, 512>(); }
TEST(BlockwiseSum, SeveralBlocksTiny)       { test_blockwise_sum<1024, 1>(); }
TEST(BlockwiseSum, SeveralBlocksMedium)     { test_blockwise_sum<1024, 32>(); }
TEST(BlockwiseSum, SeveralBlocksLarge)      { test_blockwise_sum<1024, 512>(); }
TEST(BlockwiseSum, ManyBlocksLarge)         { test_blockwise_sum<(1 << 20), 512>(); }
