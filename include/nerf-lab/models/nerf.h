#pragma once

#include <memory.h>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/object.h>

#include "nerf-lab/models/volume_renderer.h"

namespace nerf {

using json = nlohmann::json;

template<typename T>
__global__
void column_inplace_add(int n, int col, tcnn::MatrixView<float> a, const tcnn::MatrixView<float> b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a(col, i) += b(col, i);
}


template<typename T=float, typename COMPUTE_T=__half>
class SingleNetNerf : public tcnn::ObjectWithMutableHyperparams {
public:
    SingleNetNerf(const json& network) {
        config_ = std::make_unique<Config>();

        density_net_ = std::make_shared<tcnn::NetworkWithInputEncoding<__half>>(
            3,                          // n_dims_to_encode
            config_->pos_encoding_size, // n_output_dims
            network["pos_encoding"],    // encoding
            network["density_net"]      // network
        );

        dir_encoding_layer_ = std::shared_ptr<tcnn::Encoding<__half>>(
            tcnn::create_encoding<COMPUTE_T>(3, network["dir_encoding"], 0));
        
        color_net_ = std::shared_ptr<tcnn::Network<__half>>(
            tcnn::create_network<COMPUTE_T>(network["color_net"]));

        volume_renderer_layer_ = std::make_shared<VolumeRenderer<COMPUTE_T, T>>(network["num_samples_per_ray"].get<int>());

        LOG_F(INFO, "density net padded output width: %d", density_net_->padded_output_width());
        LOG_F(INFO, "color net padded output width: %d", color_net_->padded_output_width());
    }
    ~SingleNetNerf() {}

    json hyperparams() const override {
        return {{"otype", "Nerf"}};
    }

    void update_hyperparams(const json& params) override {

    }

    /**
     * @brief Forward pass of the nerf network.
     * 
     * Input positions are passed to the density mlp which returns 
     * features matrix where first row is the samples densities (usualy represented as sigma in papers),
     * the directions are passed to the dir encoding layer, and the encodings are concatenated with the
     * features, obtanied above by the density mlp. The concatenated matrix is passed to the color mlp
     * which calculates the colors of the each sample. Then the sample colors and densities are passed to the 
     * volume renderer, which calculates the color of the specific ray.
     * 
     * @param stream            cuda stream to perform the calculation in 
     * @param delta_samples     distances between the consecutive samples
     * @param input_positions   positions of the samples
     * @param input_directions  directions of the samples
     * @param output_colors     colors of the rays
     * @return std::unique_ptr<tcnn::Context> context with cached data
     */
    std::unique_ptr<tcnn::Context> forward(
        cudaStream_t stream,
        const tcnn::GPUMemory<T>& delta_samples,
        const tcnn::GPUMatrixDynamic<T>& input_positions,
        const tcnn::GPUMatrixDynamic<T>& input_directions,
        tcnn::GPUMatrix<T>* output_colors
    ) {
        int num_samples = input_positions.n();
        int num_rays = num_samples / config_->num_samples_per_ray;
        int color_net_input_size = config_->color_feature_size + config_->dir_encoding_size;

        auto ctx = std::make_unique<ForwardContext>();

        ctx->color_net_input = std::make_unique<tcnn::GPUMatrix<COMPUTE_T>>(color_net_->input_width(), num_samples);
        ctx->color_net_output = std::make_unique<tcnn::GPUMatrix<COMPUTE_T>>(color_net_->padded_output_width(), num_samples);
        
        auto density = ctx->color_net_input->slice_rows(0, 1);
        auto features = ctx->color_net_input->slice_rows(0, config_->color_feature_size);
        auto dir_encodings = ctx->color_net_input->slice_rows(
            config_->color_feature_size,
            config_->dir_encoding_size
        );
        
        // density net pass (fused encodings and mlp)
        ctx->dense_net_ctx = density_net_->forward(stream, input_positions, &features);
        
        // directional encodings pass
        ctx->dir_encoding_ctx = dir_encoding_layer_->forward(stream, input_directions, &dir_encodings);
        
        // color net pass
        cudaDeviceSynchronize();
        ctx->color_net_ctx = color_net_->forward(stream, *ctx->color_net_input, ctx->color_net_output.get());

        // rasterizer pass
        ctx->renderer_ctx = volume_renderer_layer_->forward(
            stream, delta_samples, density, *ctx->color_net_output, output_colors);

        return ctx;
    }

    /**
     * @brief Backward pass of the nerf network.
     * 
     * @param stream 
     * @param ctx 
     * @param delta_samples 
     * @param input_positions 
     * @param input_directions 
     * @param output_density_and_features 
     * @param output_colors 
     * @param dL_doutput 
     * @param dL_dinput_positions 
     * @param dL_dinput_directions 
     * @param use_inference_params 
     * @param param_gradients_mode 
     */
    void backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
        const tcnn::GPUMemory<T>& delta_samples,
		const tcnn::GPUMatrixDynamic<T>& input_positions,
        const tcnn::GPUMatrixDynamic<T>& input_directions,
		const tcnn::GPUMatrixDynamic<COMPUTE_T>& output_density_and_features,
		const tcnn::GPUMatrixDynamic<T>& output_colors,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<T>* dL_dinput_positions = nullptr,
        tcnn::GPUMatrixDynamic<T>* dL_dinput_directions = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
    ) {
        int num_samples = input_positions.n();
        int num_rays = num_samples / config_->num_samples_per_ray;
        int color_net_input_size = config_->color_feature_size + config_->dir_encoding_size;

        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

        auto density = forward.color_net_input->slice_rows(0, 1);
        auto features = forward.color_net_input->slice_rows(0, config_->color_feature_size);
        auto dir_encodings = forward.color_net_input->slice_rows(
            config_->color_feature_size,
            config_->dir_encoding_size);

        tcnn::GPUMatrix<float> dL_ddensity_addition(1, num_samples);
        tcnn::GPUMatrix<float> dL_dcolor_net_output(3, num_samples);

        volume_renderer_layer_->backward(stream, *forward.renderer_ctx, delta_samples, 
            density, *forward.color_net_output, output_colors, dL_doutput,
            &dL_ddensity_addition, &dL_dcolor_net_output,
            use_inference_params, param_gradients_mode);

        tcnn::GPUMatrix<float> dL_dcolor_net_input(color_net_input_size, num_samples);

        color_net_->backward(stream, *forward.color_net_ctx,
                                *forward.color_net_input, *forward.color_net_output, dL_dcolor_net_output,
                                &dL_dcolor_net_input,
                                use_inference_params, param_gradients_mode);

        auto dL_ddensity = dL_dcolor_net_input.slice_rows(0, 1);
        auto dL_dfeatures = dL_dcolor_net_input.slice_rows(0, config_->color_feature_size);
        auto dL_ddir_encodings = dL_dcolor_net_input.slice_rows(
            config_->color_feature_size,
            config_->dir_encoding_size);
        
        dir_encoding_layer_->backward(stream, *forward.dir_encoding_ctx,
                                        input_directions, dir_encodings, dL_ddir_encodings, 
                                        dL_dinput_directions, 
                                        use_inference_params, param_gradients_mode);

        tcnn::linear_kernel(column_inplace_add<COMPUTE_T>, 0, stream,
            num_samples,
            0,
            dL_ddensity.view(),
            dL_ddensity_addition.view()
        );

        density_net_->backward(stream, *forward.dense_net_ctx, 
                                input_positions, features, dL_dfeatures, 
                                dL_dinput_positions,
                                use_inference_params, param_gradients_mode);
    }

private:
    struct Config {
        int pos_encoding_size = 16;
        int dir_encoding_size = 16;
        int color_feature_size = 16;
        int hidden_size = 32;
        int num_samples_per_ray = 128;
    };

    struct ForwardContext : tcnn::Context {
        std::unique_ptr<tcnn::Context> dense_net_ctx;
        std::unique_ptr<tcnn::Context> dir_encoding_ctx;
        std::unique_ptr<tcnn::Context> color_net_ctx;
        std::unique_ptr<tcnn::Context> renderer_ctx;

        std::unique_ptr<tcnn::GPUMatrix<COMPUTE_T>> color_net_input;
        std::unique_ptr<tcnn::GPUMatrix<COMPUTE_T>> color_net_output;
    };

    std::shared_ptr<tcnn::NetworkWithInputEncoding<COMPUTE_T> > density_net_;
    std::shared_ptr<tcnn::Encoding<COMPUTE_T>> dir_encoding_layer_; 
    std::shared_ptr<tcnn::Network<COMPUTE_T>> color_net_;
    std::shared_ptr<VolumeRenderer<COMPUTE_T, T>> volume_renderer_layer_;

    std::unique_ptr<Config> config_;
};

template<typename T=float, typename COMPUTE_T=__half>
class Nerf : public tcnn::ObjectWithMutableHyperparams {
public:
    Nerf(const json& network) {
        config_ = std::make_unique<Config>();

        density_net_ = std::make_shared<tcnn::NetworkWithInputEncoding<__half>>(
            3,                          // n_dims_to_encode
            config_->pos_encoding_size, // n_output_dims
            network["pos_encoding"],    // encoding
            network["density_net"]      // network
        );

        dir_encoding_layer_ = std::shared_ptr<tcnn::Encoding<__half>>(
            tcnn::create_encoding<COMPUTE_T>(3, network["dir_encoding"], 0));
        
        color_net_ = std::shared_ptr<tcnn::Network<__half>>(
            tcnn::create_network<COMPUTE_T>(network["color_net"]));

        volume_renderer_layer_ = std::make_shared<VolumeRenderer<COMPUTE_T, T>>(network["num_samples_per_ray"].get<int>());

        LOG_F(INFO, "density net padded output width: %d", density_net_->padded_output_width());
        LOG_F(INFO, "color net padded output width: %d", color_net_->padded_output_width());
    }
    ~Nerf() {}

    json hyperparams() const override {
        return {{"otype", "Nerf"}};
    }

    void update_hyperparams(const json& params) override {

    }

    /**
     * @brief Forward pass of the nerf network.
     * 
     * Input positions are passed to the density mlp which returns 
     * features matrix where first row is the samples densities (usualy represented as sigma in papers),
     * the directions are passed to the dir encoding layer, and the encodings are concatenated with the
     * features, obtanied above by the density mlp. The concatenated matrix is passed to the color mlp
     * which calculates the colors of the each sample. Then the sample colors and densities are passed to the 
     * volume renderer, which calculates the color of the specific ray.
     * 
     * @param stream            cuda stream to perform the calculation in 
     * @param delta_samples     distances between the consecutive samples
     * @param input_positions   positions of the samples
     * @param input_directions  directions of the samples
     * @param output_colors     colors of the rays
     * @return std::unique_ptr<tcnn::Context> context with cached data
     */
    std::unique_ptr<tcnn::Context> forward(
        cudaStream_t stream,
        const tcnn::GPUMemory<T>& delta_samples,
        const tcnn::GPUMatrixDynamic<T>& input_positions,
        const tcnn::GPUMatrixDynamic<T>& input_directions,
        tcnn::GPUMatrix<T>* output_colors
    ) {
        int num_samples = input_positions.n();
        int num_rays = num_samples / config_->num_samples_per_ray;
        int color_net_input_size = config_->color_feature_size + config_->dir_encoding_size;

        auto ctx = std::make_unique<ForwardContext>();

        ctx->color_net_input = std::make_unique<tcnn::GPUMatrix<COMPUTE_T>>(color_net_->input_width(), num_samples);
        ctx->color_net_output = std::make_unique<tcnn::GPUMatrix<COMPUTE_T>>(color_net_->padded_output_width(), num_samples);
        
        auto density = ctx->color_net_input->slice_rows(0, 1);
        auto features = ctx->color_net_input->slice_rows(0, config_->color_feature_size);
        auto dir_encodings = ctx->color_net_input->slice_rows(
            config_->color_feature_size,
            config_->dir_encoding_size
        );
        
        // density net pass (fused encodings and mlp)
        ctx->dense_net_ctx = density_net_->forward(stream, input_positions, &features);
        
        // directional encodings pass
        ctx->dir_encoding_ctx = dir_encoding_layer_->forward(stream, input_directions, &dir_encodings);
        
        // color net pass
        cudaDeviceSynchronize();
        ctx->color_net_ctx = color_net_->forward(stream, *ctx->color_net_input, ctx->color_net_output.get());

        // rasterizer pass
        ctx->renderer_ctx = volume_renderer_layer_->forward(
            stream, delta_samples, density, *ctx->color_net_output, output_colors);

        return ctx;
    }

    /**
     * @brief Backward pass of the nerf network.
     * 
     * @param stream 
     * @param ctx 
     * @param delta_samples 
     * @param input_positions 
     * @param input_directions 
     * @param output_density_and_features 
     * @param output_colors 
     * @param dL_doutput 
     * @param dL_dinput_positions 
     * @param dL_dinput_directions 
     * @param use_inference_params 
     * @param param_gradients_mode 
     */
    void backward(
		cudaStream_t stream,
		const tcnn::Context& ctx,
        const tcnn::GPUMemory<T>& delta_samples,
		const tcnn::GPUMatrixDynamic<T>& input_positions,
        const tcnn::GPUMatrixDynamic<T>& input_directions,
		const tcnn::GPUMatrixDynamic<COMPUTE_T>& output_density_and_features,
		const tcnn::GPUMatrixDynamic<T>& output_colors,
		const tcnn::GPUMatrixDynamic<T>& dL_doutput,
		tcnn::GPUMatrixDynamic<T>* dL_dinput_positions = nullptr,
        tcnn::GPUMatrixDynamic<T>* dL_dinput_directions = nullptr,
		bool use_inference_params = false,
		tcnn::EGradientMode param_gradients_mode = tcnn::EGradientMode::Overwrite
    ) {
        int num_samples = input_positions.n();
        int num_rays = num_samples / config_->num_samples_per_ray;
        int color_net_input_size = config_->color_feature_size + config_->dir_encoding_size;

        const auto& forward = dynamic_cast<const ForwardContext&>(ctx);

        auto density = forward.color_net_input->slice_rows(0, 1);
        auto features = forward.color_net_input->slice_rows(0, config_->color_feature_size);
        auto dir_encodings = forward.color_net_input->slice_rows(
            config_->color_feature_size,
            config_->dir_encoding_size);

        tcnn::GPUMatrix<float> dL_ddensity_addition(1, num_samples);
        tcnn::GPUMatrix<float> dL_dcolor_net_output(3, num_samples);

        volume_renderer_layer_->backward(stream, *forward.renderer_ctx, delta_samples, 
            density, *forward.color_net_output, output_colors, dL_doutput,
            &dL_ddensity_addition, &dL_dcolor_net_output,
            use_inference_params, param_gradients_mode);

        tcnn::GPUMatrix<float> dL_dcolor_net_input(color_net_input_size, num_samples);

        color_net_->backward(stream, *forward.color_net_ctx,
                                *forward.color_net_input, *forward.color_net_output, dL_dcolor_net_output,
                                &dL_dcolor_net_input,
                                use_inference_params, param_gradients_mode);

        auto dL_ddensity = dL_dcolor_net_input.slice_rows(0, 1);
        auto dL_dfeatures = dL_dcolor_net_input.slice_rows(0, config_->color_feature_size);
        auto dL_ddir_encodings = dL_dcolor_net_input.slice_rows(
            config_->color_feature_size,
            config_->dir_encoding_size);
        
        dir_encoding_layer_->backward(stream, *forward.dir_encoding_ctx,
                                        input_directions, dir_encodings, dL_ddir_encodings, 
                                        dL_dinput_directions, 
                                        use_inference_params, param_gradients_mode);

        tcnn::linear_kernel(column_inplace_add<COMPUTE_T>, 0, stream,
            num_samples,
            0,
            dL_ddensity.view(),
            dL_ddensity_addition.view()
        );

        density_net_->backward(stream, *forward.dense_net_ctx, 
                                input_positions, features, dL_dfeatures, 
                                dL_dinput_positions,
                                use_inference_params, param_gradients_mode);
    }

private:
    struct Config {
        int pos_encoding_size = 16;
        int dir_encoding_size = 16;
        int color_feature_size = 16;
        int hidden_size = 32;
        int num_samples_per_ray = 128;
    };

    struct ForwardContext : tcnn::Context {
        std::unique_ptr<tcnn::Context> dense_net_ctx;
        std::unique_ptr<tcnn::Context> dir_encoding_ctx;
        std::unique_ptr<tcnn::Context> color_net_ctx;
        std::unique_ptr<tcnn::Context> renderer_ctx;

        std::unique_ptr<tcnn::GPUMatrix<COMPUTE_T>> color_net_input;
        std::unique_ptr<tcnn::GPUMatrix<COMPUTE_T>> color_net_output;
    };

    std::shared_ptr<tcnn::NetworkWithInputEncoding<COMPUTE_T> > density_net_;
    std::shared_ptr<tcnn::Encoding<COMPUTE_T>> dir_encoding_layer_; 
    std::shared_ptr<tcnn::Network<COMPUTE_T>> color_net_;
    std::shared_ptr<VolumeRenderer<COMPUTE_T, T>> volume_renderer_layer_;

    std::unique_ptr<Config> config_;
};

}