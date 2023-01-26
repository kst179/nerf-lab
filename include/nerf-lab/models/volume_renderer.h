/**
 * @file volume_renderer.h
 * @author Konstantin Kozlovtsev (kozlovtsev179@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-01-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#include <memory.h>

#include <json/json.hpp>
#include <loguru/loguru.hpp>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/object.h>

#include "nerf-lab/cuda_common.h"

namespace nerf {

using json = nlohmann::json;

template <auto START, auto END, auto INC=1, typename F>
constexpr void constexpr_for(F&& f) {
    if constexpr (START < END) {
        f(std::integral_constant<decltype(START), START>());
        constexpr_for<START + INC, END, INC>(f);
    }
}

/**
 * @brief Computes the forward pass of volume renderer
 * 
 * Lets define the \f$i\f$-th sample's density as \f$\sigma_i\f$, 
 * the distance to the next sample as \f$\Delta t_i\f$ and the color of the sample as the vector
 * \f$\mathbf{c}_i = \{r_i, g_i, b_i\}\f$
 * 
 * Then the color of the single ray is defined as (Tancik et al. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis):
 * \f$
 * \mathbf{C} = \sum_{i = 1}^N \left(
 *     T_i (1 - \exp(- \sigma_i \Delta t_i)) \mathbf{c}
 * \right)
 * \f$
 * 
 * where \f$
 * T_i = \exp\left( -\sum_{j = 1}^{i-1} \sigma_i \Delta t_i \right)
 * \f$ is visibility of the sample (how much previous )
 * 
 * let's name \f$\exp(-\sigma_i \Delta t_i)\f$ the **transparency** of the sample,
 * 
 * \f$1 - \exp(-\sigma_i \Delta t_i)\f$ the **opacity** of the sample,
 * 
 * and \f$T_i\f$ the **visibility** of the sample.  
 * 
 * Then \f$ray\_color = sample\_color \times opacity \times visibility\f$,
 * so this is what this kernel calculates.
 * 
 * @tparam NUM_SAMPLES_PER_RAY  number of samples in single ray, should be equal to blockDim
 * @param num_samples           total number of samples (= num samples per ray * num rays)
 * @param delta_samples         distance difference between samples on the ray
 * @param density               array with densities of the samples
 * @param color                 matrix with colors of the samples (rows are rgb components)
 * @param out_transparency      output array with transparancies to cache them for the backward pass
 * @param out_visibility        output array with visibilities to cache them for the backward pass
 * @param output_color          output matrix where calculated ray colors are stored (rows are rgb components)
 */
template<typename T, typename COMPUTE_T, int NUM_SAMPLES_PER_RAY>
__global__
void volume_renderer_fwd_kernel(
    int num_samples,
    const float* delta_samples,
    const tcnn::MatrixView<T> density,
    const tcnn::MatrixView<T> color,
    COMPUTE_T* out_transparency,
    COMPUTE_T* out_visibility,
    tcnn::MatrixView<COMPUTE_T> output_color
) {
    constexpr auto cumsum = blockwise_cumulative_sum<COMPUTE_T, NUM_SAMPLES_PER_RAY, true>;
    constexpr auto sum = blockwise_sum<COMPUTE_T, NUM_SAMPLES_PER_RAY>;

    assert(NUM_SAMPLES_PER_RAY == blockDim.x); // single threads block should work with samples from single ray 
                                               // (blockDim should be equal to number of samples per ray)

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;

    if (i >= num_samples) {
        return;
    }

    COMPUTE_T sigma = density(0, i);
    COMPUTE_T dt = delta_samples[i];
    COMPUTE_T minus_sigma_dt = -sigma * dt;
    COMPUTE_T transparency = exp(minus_sigma_dt);
    COMPUTE_T visibility  = exp(cumsum(minus_sigma_dt));
    COMPUTE_T sample_weight = visibility * (1 - transparency);

    COMPUTE_T rgb[3];

    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        rgb[j] = sum(sample_weight * (COMPUTE_T)color(j, i));
    }

    out_transparency[i] = transparency;
    out_visibility[i] = visibility;

    if (tid == 0) {
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            output_color(j, bid) = rgb[j];
        }
    }
}

/**
 * @brief Computes backward pass of volume renderer
 * 
 * \f$
 * \frac{\partial L}{\partial \sigma_i} = \sum_{k = 1}^3 \frac{dL}{dC^k_i} \left(
 *     T_i \exp(-\sigma_i \Delta t_i) c_i^k - 
 *     \sum_{j = i + 1}^N T_j (1 - \exp(-\sigma_i \Delta t_i)) c_j^k
 * \right) \Delta t_i
 * \f$
 * 
 * \f$
 * \frac{\partial L}{\partial c_i^k} = \frac{dL}{dC^k_i} T_i (1 - \exp(-\sigma_i \Delta t_i))
 * \f$
 * 
 * @tparam NUM_SAMPLES_PER_RAY  number of samples in single ray, should be equal to blockDim 
 * @param num_samples           total number of samples (= num samples per ray * num rays)
 * @param delta_samples         distance difference between samples on the ray     
 * @param input_density         array with densities of the samples
 * @param input_color           matrix with colors of the samples (rows are rgb components)
 * @param output_color          matrix with colors of the rays (rows are rgb components)
 * @param dL_doutput_color      matrix with derivatives wrt rays colors (same size as output_color)
 * @param transparency          array with sample transparancies cached on the forward pass
 * @param visibility            array with sample visibilities cached on the forward pass
 * @param dL_dinput_density     output matrix to store calculated derivatives wrpt densities
 * @param dL_dinput_color       output matrix to store calculated derivatives wrpt colors
 */
template<typename T, typename COMPUTE_T, int NUM_SAMPLES_PER_RAY>
__global__
void volume_renderer_bwd_kernel(
    int num_samples,
    const float* delta_samples,
    const tcnn::MatrixView<T> input_density,
    const tcnn::MatrixView<T> input_color,
    const tcnn::MatrixView<COMPUTE_T> output_color,
    const tcnn::MatrixView<COMPUTE_T> dL_doutput_color,
    const COMPUTE_T* transparency,
    const COMPUTE_T* visibility,
    tcnn::MatrixView<T> dL_dinput_density,
    tcnn::MatrixView<T> dL_dinput_color
) {
    constexpr auto reverse_cumsum = blockwise_cumulative_sum<COMPUTE_T, NUM_SAMPLES_PER_RAY, false>;

    assert(NUM_SAMPLES_PER_RAY == blockDim.x); // single threads block should work with samples from single ray 
                                               // (blockDim should be equal to number of samples per ray)   

    int ray_idx = blockIdx.x;
    int sample_idx = ray_idx * blockDim.x + threadIdx.x;

    float visibility_transparency = visibility[sample_idx] * transparency[sample_idx]; // = T_i * exp(-sigma_i * delta_i)
    float visibility_opacity = visibility[sample_idx] - visibility_transparency;       // = T_i * (1 - exp(-sigma_i * delta_i))

    #pragma unroll
    for (int color_comp = 0; color_comp < 3; ++color_comp) {
        dL_dinput_color(color_comp, sample_idx) = dL_doutput_color(color_comp, ray_idx) * visibility_opacity;
    }

    float dL_dsigma = 0;

    #pragma unroll
    for (int color_comp = 0; color_comp < 3; ++color_comp) {
        dL_dsigma += dL_doutput_color(color_comp, ray_idx) * (
            visibility_transparency * input_color(color_comp, sample_idx) - 
            reverse_cumsum(visibility_opacity * input_color(color_comp, sample_idx))
        );
    }

    dL_dsigma *= delta_samples[sample_idx];
    dL_dinput_density(0, sample_idx) = dL_dsigma;
}

template<typename T, typename COMPUTE_T=T>
class VolumeRenderer : public tcnn::ObjectWithMutableHyperparams {
public:
    VolumeRenderer(int num_samples_per_ray) {
        set_num_samples(num_samples_per_ray);
    };

    json hyperparams() const override {
        return {
            {"otype", "VolumeRenderer"},
            {"samples_per_ray", num_samples_per_ray_},
        };
    }

    void update_hyperparams(const json& params) override {
        set_num_samples(params["num_samples_per_ray"].get<int>());
    }

    void set_num_samples(int value) {
        CHECK_F(value && !(value & value - 1) && value <= 512, 
                "`num_samples_per_ray` should be a power of 2, not greater then 512, found %d", value);

        num_samples_per_ray_ = value;
    }

    int num_samples() const {
        return num_samples_per_ray_;
    }

    std::unique_ptr<tcnn::Context> forward(
		cudaStream_t stream,
        const tcnn::GPUMemory<COMPUTE_T>& delta_samples,
		const tcnn::GPUMatrixDynamic<T>& input_density,
        const tcnn::GPUMatrixDynamic<T>& input_color,
		tcnn::GPUMatrixDynamic<COMPUTE_T>* output_color
	) {
        int num_samples = input_color.n();

        auto ctx = std::make_unique<ForwardContext>();
        ctx->transparency.resize(num_samples);
        ctx->visibility.resize(num_samples);

        fwd_kernel(stream,
            num_samples,
            delta_samples.data(),
            input_density.view(),
            input_color.view(),
            ctx->transparency.data(),
            ctx->visibility.data(),
            output_color->view()
        );

        return ctx;
    }

    void backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
        const tcnn::GPUMemory<COMPUTE_T>& delta_samples,
		const tcnn::GPUMatrixDynamic<T>& input_density,
		const tcnn::GPUMatrixDynamic<T>& input_color,
		const tcnn::GPUMatrixDynamic<COMPUTE_T>& output_color,
		const tcnn::GPUMatrixDynamic<COMPUTE_T>& dL_doutput_color,
		tcnn::GPUMatrixDynamic<T>* dL_dinput_density = nullptr,
        tcnn::GPUMatrixDynamic<T>* dL_dinput_color = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
	) {
        if (dL_dinput_density == nullptr && dL_dinput_color == nullptr) {
            return;
        }

        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

        int num_samples = input_color.n();

        bwd_kernel(stream,
            num_samples,
            delta_samples.data(),
            input_density.view(),
            input_color.view(),
            output_color.view(),
            dL_doutput_color.view(),
            forward.transparency.data(),
            forward.visibility.data(),
            dL_dinput_density->view(),
            dL_dinput_color->view()
        );
    }

private:
    template<typename ...Args>
    void fwd_kernel(cudaStream_t stream, int num_samples, Args... args) {
        int threads = num_samples_per_ray_;
        int blocks = tcnn::div_round_up(num_samples, num_samples_per_ray_);

        constexpr_for<0, 10>([&](auto LOG_NUM_SAMPLES) constexpr {
            constexpr int NUM_SAMPLES = (1 << LOG_NUM_SAMPLES);
            if (num_samples_per_ray_ == NUM_SAMPLES) {
                constexpr auto kernel = volume_renderer_fwd_kernel<T, COMPUTE_T, NUM_SAMPLES>;
                kernel<<<blocks, threads, 0, stream>>>(num_samples, args...);
            }
        });
    }

    template<typename ...Args>
    void bwd_kernel(cudaStream_t stream, int num_samples, Args... args) {
        int threads = num_samples_per_ray_;
        int blocks = tcnn::div_round_up(num_samples, num_samples_per_ray_);

        constexpr_for<0, 10>([&](auto LOG_NUM_SAMPLES) constexpr {
            constexpr int NUM_SAMPLES = (1 << LOG_NUM_SAMPLES);
            if (num_samples_per_ray_ == NUM_SAMPLES) {
                constexpr auto kernel = volume_renderer_bwd_kernel<T, COMPUTE_T, NUM_SAMPLES>;
                kernel<<<blocks, threads, 0, stream>>>(num_samples, args...);
            }
        });
    }
    
    struct ForwardContext : public tcnn::Context {
        tcnn::GPUMemory<COMPUTE_T> transparency;
        tcnn::GPUMemory<COMPUTE_T> visibility;
    };

    int num_samples_per_ray_;
};

}