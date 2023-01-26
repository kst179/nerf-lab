#include <tiny-cuda-nn/common.h>
#include <loguru/loguru.hpp>

#include "nerf-lab/cuda_texture.h"

namespace nerf {

CudaTexture::CudaTexture(const cv::Mat image) {
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));

    int bpp;
    cudaChannelFormatKind channel_fmt_kind;

    if (image.type() % 8 == CV_32F) {
        bpp = sizeof(float);
        channel_fmt_kind = cudaChannelFormatKindFloat;
    } else {
        bpp = sizeof(uint8_t);
        channel_fmt_kind = cudaChannelFormatKindUnsigned;
    }

    cudaChannelFormatDesc channel_desc;
    channel_desc = cudaCreateChannelDesc(bpp, bpp, bpp, bpp, channel_fmt_kind);

    res_desc.resType = cudaResourceTypePitch2D;
    res_desc.res.pitch2D.devPtr = image.data;
    res_desc.res.pitch2D.desc = channel_desc;
    res_desc.res.pitch2D.width = image.cols;
    res_desc.res.pitch2D.height = image.rows;
    res_desc.res.pitch2D.pitchInBytes = image.cols * image.channels() * bpp;

    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.normalizedCoords = false;
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.addressMode[2] = cudaAddressModeClamp;

    CUDA_CHECK_THROW(cudaCreateTextureObject(&texture_, &res_desc, &tex_desc, nullptr));
}

CudaTexture::~CudaTexture() {
    cudaError_t error = cudaDestroyTextureObject(texture_);
    if (error != cudaSuccess) {
        ABORT_F(cudaGetErrorString(error));
    }
}

cudaTextureObject_t CudaTexture::texture_object() const {
    return texture_;
}

}