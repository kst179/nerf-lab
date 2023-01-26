#include <gtest/gtest.h>
#include <tiny-cuda-nn/common.h>
#include <random>
#include <loguru/loguru.cpp>

#include "nerf-lab/models/volume_renderer.h"

// mocking private VolumeRenderer::ForwardContext
struct ForwardContext : public tcnn::Context {
    tcnn::GPUMemory<float> transparency;
    tcnn::GPUMemory<float> visibility;
};

void assert_arrays_near(const std::vector<float>& expected, const std::vector<float>& actual, float tol = 1e-6) {
    ASSERT_EQ(expected.size(), actual.size());

    std::cout.precision(4);
    for (int i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(expected[i], actual[i], tol) << "expected[" << i << "] != actual[" << i << "] " 
                                                 << "(" << std::setw(5) << expected[i] << " != " << std::setw(5) << actual[i] << ")";
    }
}

template<typename T>
void copy_gpumat_from_host(tcnn::GPUMatrix<T>& device_array, const std::vector<T>& host_array) {
    assert(host_array.size() == device_array.m() * device_array.n());

    CUDA_CHECK_THROW(cudaMemcpy(device_array.data(), host_array.data(), 
                     host_array.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void copy_gpumat_to_host(std::vector<T>& host_array, const tcnn::GPUMatrix<T>& device_array) {
    assert(host_array.size() == device_array.m() * device_array.n());

    CUDA_CHECK_THROW(cudaMemcpy(host_array.data(), device_array.data(),
                     host_array.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

void fill_data(
    int n_samples,
    std::vector<float>& delta_samples,
    std::vector<float>& density,
    tcnn::MatrixView<float> colors,
    int seed = 179
) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution dist(0.0, 1.0);
    
    for (int i = 0; i < n_samples; ++i) {
        delta_samples[i] = 0.05 * dist(gen);
        density[i] = 5 * dist(gen);
        colors(0, i) = dist(gen);
        colors(1, i) = dist(gen);
        colors(2, i) = dist(gen);
    }
}

template<typename T>
void sequential_forward(
    int num_samples,
    int num_rays,
    const std::vector<T>& delta_samples,
    const std::vector<T>& density,
    const tcnn::MatrixView<T> colors,
    std::vector<T>& transparency,
    std::vector<T>& visibility,
    tcnn::MatrixView<T> colors_output
) {
    int samples_per_ray = num_samples / num_rays;

    for (int ray = 0; ray < num_rays; ++ray) {
        int s = ray * samples_per_ray;
        int e = s + samples_per_ray;

        visibility[s] = 0;
        for (int i = s + 1; i < e; ++i) {
            visibility[i] = visibility[i - 1] + density[i - 1] * delta_samples[i - 1];
        }

        for (int i = s; i < e; ++i) {
            visibility[i] = exp(- visibility[i]);
            transparency[i] = exp(- density[i] * delta_samples[i]);
        }

        for (int j = 0; j < 3; ++j) {
            colors_output(j, ray) = 0;

            for (int i = s; i < e; ++i) {
                colors_output(j, ray) += 
                    visibility[i] * (1 - transparency[i]) * colors(j, i);
            }
        }
    }
}

// rump-ogita-oishi alghorithm for sequential sum
void lossless_sum(double a, double b, double* s, double* t) {
    *s = a + b;
    double b_ = *s - a;
    double a_ = *s - b;
    *t = (a - a_) + (b - b_);
}

double l2_loss(const std::vector<double>& data) {
    double loss = 0;
    double tail = 0;

    for (int i = 0; i < data.size(); ++i) {
        lossless_sum(loss, data[i] * data[i] + tail, &loss, &tail);
    }

    return loss;
}

TEST(TestVolumeRenderer, SimpleForward) {
    constexpr int N_SAMPLES = 1024;
    constexpr int SPR = 128; // samples per ray
    constexpr int N_RAYS = N_SAMPLES / SPR;

    nerf::VolumeRenderer<float> renderer(SPR);

    std::vector<float> delta_samples(N_SAMPLES);
    std::vector<float> density(N_SAMPLES);
    std::vector<float> colors_data(3 * N_SAMPLES);

    std::vector<float> ray_colors_expected_data(3 * N_RAYS);
    std::vector<float> ray_colors_actual_data(3 * N_RAYS);

    std::vector<float> transparency_expected(N_SAMPLES);
    std::vector<float> transparency_actual(N_SAMPLES);

    std::vector<float> visibility_expected(N_SAMPLES);
    std::vector<float> visibility_actual(N_SAMPLES);

    tcnn::MatrixView colors(&colors_data[0], 1, 3);
    tcnn::MatrixView ray_colors_expected(&ray_colors_expected_data[0], 1, 3);

    fill_data(N_SAMPLES, delta_samples, density, colors);

    sequential_forward(
        N_SAMPLES, N_RAYS,
        delta_samples, density, colors,
        transparency_expected, visibility_expected, ray_colors_expected
    );
    
    tcnn::GPUMemory<float> delta_samples_gpu(N_SAMPLES);
    tcnn::GPUMatrix<float> density_gpu(1, N_SAMPLES);
    tcnn::GPUMatrix<float> colors_gpu(3, N_SAMPLES);
    tcnn::GPUMatrix<float> ray_colors_gpu(3, N_RAYS);

    delta_samples_gpu.copy_from_host(delta_samples);
    copy_gpumat_from_host(density_gpu, density);
    copy_gpumat_from_host(colors_gpu, colors_data);

    auto ctx = renderer.forward(nullptr, delta_samples_gpu, density_gpu, colors_gpu, &ray_colors_gpu);
    const ForwardContext* forward = (const ForwardContext*)(ctx.get());    

    cudaDeviceSynchronize();

    forward->transparency.copy_to_host(transparency_actual);
    forward->visibility.copy_to_host(visibility_actual);
    copy_gpumat_to_host(ray_colors_actual_data, ray_colors_gpu);

    assert_arrays_near(transparency_expected, transparency_actual);
    assert_arrays_near(visibility_expected, visibility_actual);
    assert_arrays_near(ray_colors_expected_data, ray_colors_actual_data);
}

TEST(TestVolumeRenderer, SimpleBackward) {
    constexpr int N_SAMPLES = 1024;
    constexpr int SPR = 128; // samples per ray
    constexpr int N_RAYS = N_SAMPLES / SPR;
    constexpr double eps = 1e-8;

    nerf::VolumeRenderer<float> renderer(SPR);

    // initial arrays
    std::vector<float> delta_samples(N_SAMPLES);
    std::vector<float> density(N_SAMPLES);
    std::vector<float> colors_data(3 * N_SAMPLES);
    std::vector<float> ray_colors_data(3 * N_RAYS);

    std::vector<float> transparency(N_SAMPLES);
    std::vector<float> visibility(N_SAMPLES);

    tcnn::MatrixView colors(colors_data.data(), 1, 3);
    tcnn::MatrixView ray_colors(ray_colors_data.data(), 1, 3);

    // same in doubles

    std::vector<double> delta_samples_dbl(N_SAMPLES);
    std::vector<double> density_dbl(N_SAMPLES);
    std::vector<double> colors_data_dbl(3 * N_SAMPLES);
    std::vector<double> ray_colors_data_dbl(3 * N_RAYS);

    std::vector<double> transparency_dbl(N_SAMPLES);
    std::vector<double> visibility_dbl(N_SAMPLES);

    tcnn::MatrixView colors_dbl(colors_data_dbl.data(), 1, 3);
    tcnn::MatrixView ray_colors_dbl(ray_colors_data_dbl.data(), 1, 3);

    // backward input derivative 

    std::vector<float> dL_dray_colors_data(3 * N_RAYS);

    // output derivatives

    std::vector<float> dL_ddensity_expected(N_SAMPLES);
    std::vector<float> dL_ddensity_actual(N_SAMPLES);

    std::vector<float> dL_dcolors_expected_data(3 * N_SAMPLES);
    std::vector<float> dL_dcolors_actual_data(3 * N_SAMPLES);

    tcnn::MatrixView<float> dL_dray_colors(dL_dray_colors_data.data(), 1, 3);
    tcnn::MatrixView<float> dL_dcolors_expected(dL_dcolors_expected_data.data(), 1, 3);


    fill_data(N_SAMPLES, delta_samples, density, colors);

    for (int i = 0; i < N_SAMPLES; ++i) {
        delta_samples_dbl[i] = delta_samples[i];
        density_dbl[i] = density[i];

        colors_dbl(0, i) = colors(0, i);
        colors_dbl(1, i) = colors(1, i);
        colors_dbl(2, i) = colors(2, i);
    }

    // standard forward (in floats)
    sequential_forward(
        N_SAMPLES, N_RAYS,
        delta_samples, density, colors,
        transparency, visibility, ray_colors
    );

    for (int i = 0; i < ray_colors_data.size(); ++i) {
        ray_colors_data_dbl[i] = ray_colors_data[i];
        dL_dray_colors_data[i] = 2 * ray_colors_data[i];
    }

    // numerical backward (in doubles for higher precision)
    for (int i = 0; i < N_SAMPLES; ++i) {
        float tmp = density[i];

        density_dbl[i] = tmp - eps;

        sequential_forward(
            N_SAMPLES, N_RAYS,
            delta_samples_dbl, density_dbl, colors_dbl,
            transparency_dbl, visibility_dbl, ray_colors_dbl
        );

        double f0 = l2_loss(ray_colors_data_dbl);

        density_dbl[i] = tmp + eps;

        sequential_forward(
            N_SAMPLES, N_RAYS,
            delta_samples_dbl, density_dbl, colors_dbl,
            transparency_dbl, visibility_dbl, ray_colors_dbl
        );

        double f1 = l2_loss(ray_colors_data_dbl);

        dL_ddensity_expected[i] = (f1 - f0) / (2 * eps);

        density_dbl[i] = tmp;
    }

    for (int i = 0; i < N_SAMPLES; ++i) {
        for (int j = 0; j < 3; ++j) {
            float tmp = colors_dbl(j, i);
            
            colors_dbl(j, i) = tmp - eps;

            sequential_forward(
                N_SAMPLES, N_RAYS,
                delta_samples_dbl, density_dbl, colors_dbl,
                transparency_dbl, visibility_dbl, ray_colors_dbl
            );

            double f0 = l2_loss(ray_colors_data_dbl);

            colors_dbl(j, i) = tmp + eps;

            sequential_forward(
                N_SAMPLES, N_RAYS,
                delta_samples_dbl, density_dbl, colors_dbl,
                transparency_dbl, visibility_dbl, ray_colors_dbl
            );

            double f1 = l2_loss(ray_colors_data_dbl);
            dL_dcolors_expected(j, i) = (f1 - f0) / (2 * eps);

            colors_dbl(j, i) = tmp;
        }
    }

    // move float arrays to gpu
    tcnn::GPUMemory<float> delta_samples_gpu(N_SAMPLES);
    tcnn::GPUMatrix<float> density_gpu(1, N_SAMPLES);
    tcnn::GPUMatrix<float> colors_gpu(3, N_SAMPLES);
    tcnn::GPUMatrix<float> ray_colors_gpu(3, N_RAYS);
    tcnn::GPUMatrix<float> dL_dray_colors_gpu(3, N_RAYS);

    tcnn::GPUMatrix<float> dL_ddensity_gpu(1, N_SAMPLES);
    tcnn::GPUMatrix<float> dL_dcolors_gpu(3, N_SAMPLES);

    delta_samples_gpu.copy_from_host(delta_samples);
    copy_gpumat_from_host(density_gpu, density);
    copy_gpumat_from_host(colors_gpu, colors_data);
    copy_gpumat_from_host(ray_colors_gpu, ray_colors_data);
    copy_gpumat_from_host(dL_dray_colors_gpu, dL_dray_colors_data);

    auto ctx = renderer.forward(nullptr, 
        delta_samples_gpu, density_gpu, colors_gpu,
        &ray_colors_gpu
    );

    renderer.backward(nullptr, *ctx,
        delta_samples_gpu, density_gpu, colors_gpu, ray_colors_gpu, dL_dray_colors_gpu,
        &dL_ddensity_gpu, &dL_dcolors_gpu
    );

    cudaDeviceSynchronize();

    copy_gpumat_to_host(dL_ddensity_actual, dL_ddensity_gpu);
    copy_gpumat_to_host(dL_dcolors_actual_data, dL_dcolors_gpu);

    assert_arrays_near(dL_ddensity_expected, dL_ddensity_actual);
    assert_arrays_near(dL_dcolors_expected_data, dL_dcolors_actual_data);
}