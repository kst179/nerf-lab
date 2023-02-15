#include <gtest/gtest.h>

#include <mma.h>
#include <cstdint>
#include <chrono>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <nerf-lab/models/qat_fully_fused_mlp.h>

using namespace tcnn;
using namespace nerf;

template <typename T>
__global__ void fill_rand(uint32_t size, T* out, tcnn::default_rng_t rng) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (i >= size) {
        return;
    }

    rng.advance(i);

    if (std::is_same<T, __half>::value) {
        out[i] = (T)((rng.next_float() - 0.5) * 2);
    } else {
        out[i] = (T)rng.next_uint();
    }
}

__global__ void copy_weights_with_pad(
    uint32_t width,
    uint32_t padded_width,
    uint32_t* __restrict__ padded_weights, 
    const uint32_t* __restrict__ weights
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int col = i % padded_width;
    int row = i / padded_width;

    padded_weights[i] = (col < width) ? weights[row * width + col] : 0;
}

template <
    typename T, 
    typename T_ST = storage_t<T>,
    typename T_ACC = accumulator_t<T>
>
__global__ void check(uint32_t size, T_ACC* expected, T_ST* actual);

template <>
__global__ void check<int8_t, int8_t, int32_t>(uint32_t size, int32_t* expected, int8_t* actual) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) {
        return;
    }

    int8_t x = expected[i];
    int8_t y = actual[i];

    if (x != y) {
        // printf("%d %4d %d != %d\n", size, i, x, y);
        assert(false);
    }
}

template <>
__global__ void check<__half, __half, __half>(uint32_t size, __half* expected, __half* actual) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) {
        return;
    }

    float x = expected[i];
    float y = actual[i];

    if (abs(x - y) > 5e-2) {
        // printf("%4d %f != %f\n", i, x, y);
        assert(false);
    }
}

template <>
__global__ void check<uint1b_t, uint32_t, int32_t>(uint32_t size, int32_t* expected, uint32_t* actual) {
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= size) {
        return;
    }

    for (int k = 0; k < 32; ++k) {
        int x = expected[i * 32 + k] & 1;
        int y = (actual[i] >> k) & 1;

        if (x != y) {
            // printf("%4d %f != %f\n", i, x, y);
            assert(false);
        }
    }
}

template <int WIDTH>
__global__ void naive_bmm(
    const uint32_t* __restrict__ in, 
    const uint32_t* __restrict__ weights, 
    int32_t* __restrict__ out
) {
    constexpr int ST_WIDTH = WIDTH / 32;
    
    int row = blockIdx.x;
    int weights_col = threadIdx.x;

    int32_t y = 0;
    
    for (int k = 0; k < ST_WIDTH; ++k) {
        uint32_t x = in[row * ST_WIDTH + k];
        uint32_t w = weights[k + weights_col * ST_WIDTH];

        y += __popc(x ^ w);
    }

    out[row * WIDTH + weights_col] = y;
}

template <
    typename CalculationPolicy, 
    typename T = typename CalculationPolicy::mm_type,
    typename T_ST = typename CalculationPolicy::storage_type, 
    typename T_ACC = typename CalculationPolicy::accumulator_type
>
__global__ void test_kernel(
    const T_ST* __restrict__ input, 
    const T_ST* __restrict__ weights, 
    T_ST* __restrict__ output,
    int n_layers = 1
) {
    using _ = CalculationPolicy;
    
    const uint32_t chunk_idx = blockIdx.x;
    const uint32_t chunk_offset = chunk_idx * _::storage_chunk_size;

    extern __shared__ uint8_t shmem[];
    T_ST* act_shmem = (T_ST*)shmem;
    T_ACC* aux_shmem = (T_ACC*)((T_ST*)shmem + _::shmem_size);

    if (_::partial_tc_load) {
        fill_zeroes<CalculationPolicy>(act_shmem);
    }

    threadblock_load_input_static<CalculationPolicy>(act_shmem, input + chunk_offset);

    #pragma unroll
    for (int i = 0; i < n_layers; ++i) {
        qat_threadblock_layer<CalculationPolicy, false, T, T_ST, T_ACC>(act_shmem, weights, nullptr, nullptr, aux_shmem);
    }

    threadblock_store_output_static<CalculationPolicy>(output + chunk_offset, act_shmem);
}

template <typename CalculationPolicy>
std::enable_if_t< std::is_same_v<typename CalculationPolicy::mm_type, __half> || 
                  std::is_same_v<typename CalculationPolicy::mm_type, int8_t> > 
test(uint32_t batch_size) {
    using _ = CalculationPolicy;
    using T = typename CalculationPolicy::mm_type;
    using T_ST = typename CalculationPolicy::storage_type;
    using T_ACC = typename CalculationPolicy::accumulator_type;

    // Using float accumulator for cutlass multiplication, 
    // cause __half's reduction sum leads to enormous numerical errors
    using T_ACC_CUTLASS = std::conditional_t<std::is_same_v<T, __half>, float, T_ACC>;

    using GemmOp = cutlass::gemm::device::Gemm<
        /*ElementA =*/ T,
        /*LayoutA =*/ cutlass::layout::RowMajor,
        /*ElementB =*/ T,
        /*LayoutB =*/ cutlass::layout::ColumnMajor,
        /*ElementC =*/ T_ACC,
        /*LayoutC =*/ cutlass::layout::RowMajor,
        /*ElementAccumulator = */ T_ACC_CUTLASS
    >;

    GPUMatrix<T, MatrixLayout::RowMajor> input(batch_size, _::chunk_width);
    GPUMatrix<T, MatrixLayout::ColumnMajor> weights(_::chunk_width, _::chunk_width);
    GPUMatrix<T_ACC, MatrixLayout::RowMajor> output_expected(batch_size, _::chunk_width);
    GPUMatrix<T, MatrixLayout::RowMajor> output_actual(batch_size, _::storage_width);

    default_rng_t rng{1337};

    rng.advance();
    linear_kernel(fill_rand<T>, 0, nullptr, input.n_elements(), input.data(), rng);

    rng.advance();
    linear_kernel(fill_rand<T>, 0, nullptr, weights.n_elements(), weights.data(), rng);

    GemmOp gemm_op;
    cutlass::Status status;
    
    status = gemm_op({
        {(int)batch_size, (int)_::chunk_width, (int)_::chunk_width},
        {input.data(), (int)input.stride()},
        {weights.data(), (int)weights.stride()},
        {output_expected.data(), (int)output_expected.stride()},
        {output_expected.data(), (int)output_expected.stride()},
        {1, 0}
    });

    ASSERT_EQ(status, cutlass::Status::kSuccess) << "Got cutlass error: " << cutlass::cutlassGetStatusString(status);

    assert(batch_size % _::chunk_height == 0);

    dim3 threads { 32, _::n_warps, 1 };
    dim3 blocks { batch_size / _::chunk_height, 1, 1 };

    int shmem_size_bytes = _::shmem_size * sizeof(T) + 
                           (std::is_same<T, T_ACC>::value ? 0 : _::acc_shmem_size * sizeof(T_ACC));

    ASSERT_NO_THROW(CUDA_CHECK_THROW(
        cudaFuncSetAttribute(test_kernel<CalculationPolicy>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_bytes)));

    test_kernel<CalculationPolicy><<<blocks, threads, shmem_size_bytes>>>(input.data(), weights.data(), output_actual.data());
    ASSERT_NO_THROW(CUDA_CHECK_THROW(cudaDeviceSynchronize()));

    linear_kernel(check<T, T_ST, T_ACC>, 0, nullptr, output_expected.n_elements(), output_expected.data(), output_actual.data());
 
    ASSERT_NO_THROW(CUDA_CHECK_THROW(cudaDeviceSynchronize()));

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template <typename CalculationPolicy>
std::enable_if_t< std::is_same_v<typename CalculationPolicy::mm_type, uint1b_t> > 
test(uint32_t batch_size) {
    using _ = CalculationPolicy;
    using T = typename CalculationPolicy::mm_type;
    using T_ST = typename CalculationPolicy::storage_type;
    using T_ACC = typename CalculationPolicy::accumulator_type;

    GPUMatrix<T_ST, MatrixLayout::RowMajor> input(batch_size, _::storage_width);
    GPUMatrix<T_ST, MatrixLayout::ColumnMajor> weights(_::storage_width, _::chunk_width);
    GPUMatrix<T_ST, MatrixLayout::ColumnMajor> padded_weights(_::padded_storage_width, _::chunk_width);
    GPUMatrix<T_ACC, MatrixLayout::RowMajor> output_expected(batch_size, _::chunk_width);
    GPUMatrix<T_ST, MatrixLayout::RowMajor> output_actual(batch_size, _::storage_width);

    default_rng_t rng{1337};

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, input.n_elements(), input.data(), rng);

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, weights.n_elements(), weights.data(), rng);

    copy_weights_with_pad<<<batch_size * _::padded_storage_width / 128, 128>>>(_::storage_width, _::padded_storage_width, padded_weights.data(), weights.data());

    assert(batch_size % _::chunk_height == 0);

    {
        int threads = _::chunk_width;
        int blocks = batch_size;
        naive_bmm<_::chunk_width><<<blocks, threads>>>(input.data(), weights.data(), output_expected.data());
        ASSERT_NO_THROW(CUDA_CHECK_THROW(cudaDeviceSynchronize()));
    }

    {
        dim3 threads { 32, _::n_warps, 1 };
        dim3 blocks { batch_size / _::chunk_height, 1, 1 };

        int shmem_size_bytes = _::shmem_size * sizeof(T_ST) + 
                            (std::is_same<T, T_ACC>::value ? 0 : _::acc_shmem_size * sizeof(T_ACC));

        ASSERT_NO_THROW(CUDA_CHECK_THROW(cudaFuncSetAttribute(test_kernel<CalculationPolicy>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_bytes)));
        test_kernel<CalculationPolicy><<<blocks, threads, shmem_size_bytes>>>(input.data(), padded_weights.data(), output_actual.data());
        ASSERT_NO_THROW(CUDA_CHECK_THROW(cudaDeviceSynchronize()));
    }

    linear_kernel(check<T, T_ST, T_ACC>, 0, nullptr, output_actual.n_elements(), output_expected.data(), output_actual.data());
 
    ASSERT_NO_THROW(CUDA_CHECK_THROW(cudaDeviceSynchronize()));

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

template <typename CalculationPolicy>
void bench(uint32_t batch_size) {
    using _ = CalculationPolicy;
    using T = typename CalculationPolicy::mm_type;
    using T_ST = typename CalculationPolicy::storage_type;
    using T_ACC = typename CalculationPolicy::accumulator_type;

    GPUMatrix<T_ST, MatrixLayout::RowMajor> input(batch_size, _::storage_width);
    GPUMatrix<T_ST, MatrixLayout::ColumnMajor> weights(_::storage_width, _::chunk_width);
    GPUMatrix<T_ST, MatrixLayout::RowMajor> output(batch_size, _::storage_width);

    default_rng_t rng{1337};

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, input.n_elements(), input.data(), rng);

    rng.advance();
    linear_kernel(fill_rand<T_ST>, 0, nullptr, weights.n_elements(), weights.data(), rng);

    assert(batch_size % _::chunk_height == 0);

    dim3 threads { 32, _::n_warps, 1 };
    dim3 blocks { batch_size / _::chunk_height, 1, 1 };

    int shmem_size_bytes = _::shmem_size * sizeof(T_ST) + 
                           (std::is_same<T, T_ACC>::value ? 0 : _::acc_shmem_size * sizeof(T_ACC));

    CUDA_CHECK_THROW(cudaFuncSetAttribute(test_kernel<CalculationPolicy>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_bytes));
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    
    constexpr int n_batches = 100;

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n_batches; ++i) {
        test_kernel<CalculationPolicy><<<blocks, threads, shmem_size_bytes>>>(input.data(), weights.data(), output.data(), 8);
    }

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    std::cout << std::setw(10) << (float)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1'000'000 / n_batches << " ms/batch" << std::endl;
}

template <uint32_t width> using HalfTestPolicy = FulllyFusedMatMulCalculationPolicy<
    /*T=*/ __half,
    /*WIDTH=*/ width,
    /*HEIGHT=*/ 128,
    /*N_WARPS_COL=*/ width / 16,
    /*N_WARPS_ROW=*/ 1,
    /*T_COPY=*/ uint4
>;

template <uint32_t width> using Int8TestPolicy = FulllyFusedMatMulCalculationPolicy<
    /*T=*/ int8_t,
    /*WIDTH=*/ width,
    /*HEIGHT=*/ 128,
    /*N_WARPS_COL=*/ width / 16,
    /*N_WARPS_ROW=*/ 1,
    /*T_COPY=*/ uint4
>;

template <uint32_t width> using BitTestPolicy = FulllyFusedMatMulCalculationPolicy<
    /*T=*/ uint1b_t,
    /*WIDTH=*/ width,
    /*HEIGHT=*/ 128,
    /*N_WARPS_COL=*/ width / 8 < 8 ? width / 8 : 8,
    /*N_WARPS_ROW=*/ 1,
    /*T_COPY=*/ uint32_t
>;

TEST(QATThreadblockForward, Half16) { test<HalfTestPolicy<16>>(1024); }
TEST(QATThreadblockForward, Half32) { test<HalfTestPolicy<32>>(1024); }
TEST(QATThreadblockForward, Half64) { test<HalfTestPolicy<64>>(1024); }
TEST(QATThreadblockForward, Half128) { test<HalfTestPolicy<128>>(1024); }

TEST(QATThreadblockForward, Int8x16) { test<Int8TestPolicy<16>>(1024); }
TEST(QATThreadblockForward, Int8x32) { test<Int8TestPolicy<32>>(1024); }
TEST(QATThreadblockForward, Int8x64) { test<Int8TestPolicy<64>>(1024); }
TEST(QATThreadblockForward, Int8x128) { test<Int8TestPolicy<128>>(1024); }

TEST(QATThreadblockForward, Bit32) { test<BitTestPolicy<32>>(1024); }
TEST(QATThreadblockForward, Bit64) { test<BitTestPolicy<64>>(1024); }
TEST(QATThreadblockForward, Bit128) { test<BitTestPolicy<128>>(1024); }
