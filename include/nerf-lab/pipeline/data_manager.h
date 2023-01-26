#pragma once

#include <vector>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/random.h>

#include "nerf-lab/data/dataset.h"
#include "nerf-lab/cuda_texture.h"

namespace nerf {

/**
 * @brief Structure with single image metadata.
 * 
 * Contains camera intrinsics, extrinsics, image size and offset in image data array.
 */
struct ImageMeta {
    /** width of the image */
    int width;
    /** height of the image */
    int height;
    /** offset in the image data array (in floats) */
    int offset;
    /** extrinsics matrix (camera transform) */
    Isometry3f transform;
    /** camera focal distance (scale params of intrinsic matrix) */
    Vector2f focal_dist;
    /** camera center point (translation param of intrinsic matrix) */
    Vector2f center_point;
    /** distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6) according to OpenCV distortion model */
    float dist_coefs[8];
    /** distance from origin to ray - near plane intersection */
    float near;
    /** distance from origin to ray - far plane intersection */
    float far;
};

/**
 * @brief Structure with single camera ray parameters
 * 
 * Contains information about ray geometry and corresponding pixel color
 */
struct Ray {
    /** color of the pixel corresponding to the ray */
    Vector4f color;
    /** origin of the ray (point in 3d space) */
    Vector3f origin;
    /** direction of the ray (unit 3d vector) */
    Vector3f dir;
    /** distance to near camera's plane */
    float near;
    /** distance to far camera's plane */
    float far;
};

/**
 * @brief Class, responsible for rays and samples generation from dataset
 * 
 */
class DataManager {
public:
    /**
     * @brief Construct a new Data Manager object
     * 
     * @param dataset dataset which contains images info and camera parameters
     * @param box_transform transform to unit cube in which points would be sampled
     * @param near near distance in camera model (used to sample only visible points)
     * @param far far distance in camera model
     */
    DataManager(
        const std::shared_ptr<ImagesDataset> dataset, 
        Isometry3f box_transform, 
        float near = 0.01,
        float far = 100.0
    );

    /**
     * @brief Loads all images to inner gpu memory
     */
    void load_images();

    /**
     * @brief Loads some images to inner gpu memory
     * 
     * @param image_ids list of images to be loaded
     */
    void load_images(const std::vector<int>& image_ids);

    /**
     * @brief Samples rays to inner gpu memory
     * 
     * @param stream cuda stream to run kernels in
     * @param num_rays number of rays to sample
     */
    void sample_rays(cudaStream_t stream, int num_rays);

    /**
     * @brief Samples points on the previosly sampled rays
     * 
     * @param stream cuda stream to run kernels in
     * @param num_samples_per_ray number of samples per one ray
     * @param[out] points output matrix to store sampled points 
     * @param[out] dirs output matrix to store view directions
     */
    void sample_points(cudaStream_t stream, int num_samples_per_ray, 
                       tcnn::GPUMatrix<float>& points, tcnn::GPUMatrix<float>& dirs);

    /**
     * @brief Frees all gpu memory
     */
    void free_gpu();

private:
    tcnn::default_rng_t rng_;

    tcnn::GPUMemory<ImageMeta> images_meta_;
    tcnn::GPUMemory<float> images_data_;
    tcnn::GPUMemory<Ray> rays_;
    tcnn::GPUMemory<Isometry3f> box_transform_;

    std::shared_ptr<ImagesDataset> dataset_;

    // Isometry3f box_transform_;
    float near_;
    float far_;
};

}