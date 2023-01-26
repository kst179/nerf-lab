#include <loguru/loguru.hpp>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include "nerf-lab/pipeline/nerf_trainer.h"
#include "nerf-lab/models/nerf.h"

#undef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(x)                                                                                               \
	do {                                                                                                                  \
		cudaError_t result = x;                                                                                           \
		if (result != cudaSuccess) {                                                                                     \
            LOG_F(ERROR, "CUDA error: %s", cudaGetErrorString(result));                                                  \
			throw std::runtime_error(std::string(FILE_LINE " " #x " failed with error ") + cudaGetErrorString(result));  \
        }                                                                                                                \
	} while(0)

int main(int argc, char** argv) {
    loguru::init(argc, argv);
    loguru::set_fatal_handler([](const loguru::Message& msg) {
        cudaError_t error = cudaGetLastError();
        if (error) {
            const char* name = cudaGetErrorName(error);
            const char* message = cudaGetErrorString(error);

            LOG_F(ERROR, "Last cuda error: %s - %s", name, message);
        }
        
        throw std::runtime_error(std::string(msg.prefix) + msg.message);
    });

    nerf::json config = {
        {"pos_encoding", {
            {"otype", "HashGrid"},
            {"n_levels", 16},
            {"n_features_per_level", 2},
            {"log2_hashmap_size", 20},
            {"base_resolution", 16},
            {"per_level_scale", 2.0},
        }},
        {"dir_encoding", {
            {"otype", "SphericalHarmonics"},
            {"degree", 4},
        }},
        {"density_net", {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "ReLU"},
            {"n_input_dims", 16},
            {"n_output_dims", 16},
            {"n_neurons", 16},
            {"n_hidden_layers", 1},
        }},
        {"color_net", {
            {"otype", "FullyFusedMLP"},
            {"activation", "ReLU"},
            {"output_activation", "Sigmoid"},
            {"n_input_dims", 32},
            {"n_output_dims", 3},
            {"n_neurons", 16},
            {"n_hidden_layers", 1},
        }},
        {"num_samples_per_ray", 128},
    };

    auto nerf_model = std::make_shared<nerf::Nerf<float, __half>>(config);

    int n = 1024;
    tcnn::GPUMemory<float> delta_samples(n);
    tcnn::GPUMatrix<float> input_positions(3, n);
    tcnn::GPUMatrix<float> input_directions(3, n);
    tcnn::GPUMatrix<float> output_colors(3, n / 128);

    try {
        nerf_model->forward(
         nullptr, delta_samples, input_positions, input_directions, &output_colors);
    } catch (std::runtime_error) {
        cudaError_t error = cudaGetLastError();
        if (error) {
            const char* name = cudaGetErrorName(error);
            const char* message = cudaGetErrorString(error);

            LOG_F(ERROR, "Last cuda error: %s - %s", name, message);
        }
    }

    return 0;
}