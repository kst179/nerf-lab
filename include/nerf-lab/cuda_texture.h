#include <opencv2/core.hpp>

namespace nerf {

class CudaTexture {
public:
    CudaTexture(const cv::Mat image);
    ~CudaTexture();
    CudaTexture(const CudaTexture& other) = delete;
    CudaTexture operator=(const CudaTexture& other) = delete;

    cudaTextureObject_t texture_object() const;

private:
    cudaTextureObject_t texture_;
};

}