#include <cuda/std/functional>
#include <tiny-cuda-nn/common.h>
#include <Eigen/Core>
#include <opencv2/imgproc.hpp>
#include <loguru/loguru.hpp>

#include "nerf-lab/pipeline/data_manager.h"
#include "nerf-lab/data/image_handle.h"

namespace nerf {

using namespace Eigen;

constexpr int NUM_CHANNELS = 4; 

__device__ 
void undistort_point(
    const ImageMeta* __restrict__ meta,
    float &x,
    float &y,
    int n_iters = 5
) {
    // convert to normalized coords
    x = (x - meta->center_point.x()) / meta->focal_dist.x();
    y = (y - meta->center_point.y()) / meta->focal_dist.y();

    // iterative undistortion
    float x0 = x, y0 = y;
    const float* k = &meta->dist_coefs[0];

    for(int j = 0; j < n_iters; j++) {
        float r2 = x*x + y*y;
        float icdist = (1 + ((k[7]*r2 + k[6])*r2 + k[5])*r2)/(1 + ((k[4]*r2 + k[1])*r2 + k[0])*r2);
        float deltaX = 2*k[2]*x*y + k[3]*(r2 + 2*x*x);
        float deltaY = k[2]*(r2 + 2*y*y) + 2*k[3]*x*y;
        x = (x0 - deltaX)*icdist;
        y = (y0 - deltaY)*icdist;
    }
}

__device__ Array4f image_at(
    int i, int j, 
    const float* __restrict__ image,
    const ImageMeta* __restrict__ meta) {

    i = (i * meta->width + j) * NUM_CHANNELS;
    return Array4f(
        image[i+0],
        image[i+1],
        image[i+2],
        image[i+3]
    );
}

__device__
Array4f linear_interpolaiton(
    float x, float y,
    const float* __restrict__ image,
    const ImageMeta* __restrict__ meta
) {
    int i = floor(x);
    int j = floor(y);

    if (i < 0 || j < 0 || i + 1 >= meta->width || j + 1 >= meta->height) {
        return Array4f::Zero();
    }

    float dx = x - i;
    float dy = y - j;

    float ix = 1 - dx;
    float iy = 1 - dy;

    Array4f color = image_at(  i,   j, image, meta) * ix * iy + 
                    image_at(i+1,   j, image, meta) * dx * iy +
                    image_at(  i, j+1, image, meta) * ix * dy + 
                    image_at(i+1, j+1, image, meta) * dx * dy;

    return color;
}

/**
 * @brief Calculates ray to box intersections.
 * 
 * Finds two interesection points of ray and box. Ray is represented as point + direction, and box represented as 
 * a 3d transform (to the local space where cube corners have modulo 1 coordinates).
 * Then crops it by near/far parameters and returns as two distances, which defines two points on the ray.
 * 
 * @param o             ray origin
 * @param d             ray direction (should be unit vector)
 * @param box_transform affine transform to the space where box has unit coordinates (from -1 to 1)
 * @param near          distance from origin to the closest point which could be a sample
 * @param far           distance from origin to the most distinct point which could be a sample 
 * @param [out] start   distance to the start of the segment where samples can be located
 * @param [out] end     distance to the end of the segment where samples can be located 
 */
__device__
void box_intersections(
    Vector3f o,
    Vector3f d,
    const Isometry3f* box_transform,
    float near,
    float far,
    float& start,
    float& end
) {
    static constexpr float eps = 1e-6;

    // transform ray to box coordinates (where box is unit box: [0, 1]^3)
    o = *box_transform * o;
    d = *box_transform * d;

    // calculate intersections to 6 box's planes
    float t[6];

    t[0] = abs(d.x()) > eps ? (-1.0 - o.x()) / d.x() : -1;
    t[1] = abs(d.x()) > eps ? ( 1.0 - o.x()) / d.x() : -1;

    t[2] = abs(d.y()) > eps ? (-1.0 - o.y()) / d.y() : -1;
    t[3] = abs(d.y()) > eps ? ( 1.0 - o.y()) / d.y() : -1;

    t[4] = abs(d.z()) > eps ? (-1.0 - o.z()) / d.z() : -1;
    t[5] = abs(d.z()) > eps ? ( 1.0 - o.z()) / d.z() : -1;

    // select two of them, which are on the box surface
    float t0 = -1;
    float t1 = -1;
    Vector3f p;

    for (int i = 0; i < 6; ++i) {
        // point is on cube if max(x, y, z) = 1
        p = o + t[i] * d;
        float max_coord = max( abs(p.x()), 
                          max( abs(p.y()), 
                               abs(p.z()) ) );
        if (abs(max_coord - 1.0) <= eps) {
            if (t0 != -1) {
                t0 = t[i];
            } else {
                t1 = t[i];
            }
        }
    }

    // no actual intersections found
    if (t0 == -1 && t1 == -1) {
        return;
    }

    // swap to make always `t0 < t1`
    if (t0 > t1) {
        float tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    
    // if first intersection is behind the origin or too close to it, 
    // then move it to `near` (slightly after it)
    if (t0 < near) {
        t0 = near;
    }

    if (t1 > far) {
        t1 = far;
    }
}

/**
 * @brief Samples rays from given cameras.
 * 
 * @param num_images    number of images from which rays are sampled
 * @param num_rays      number of rays to sample
 * @param images_meta   array of images metadata (intrinsics, extrinsics, etc.) 
 * @param images_data   images data: float rgba array, row-wise, channels-last, stacked onto each other
 * @param box_transform transform of nerf's bounding box, where points should be sampled
 * @param [out] rays    rays array, to store sampled rays
 * @param rng           random generator
 */
__global__
void sample_rays_kernel(
    int num_rays,
    int num_images,
    float near,
    float far,
    const ImageMeta* __restrict__ images_meta,
    const float* __restrict__ images_data,
    const Isometry3f* __restrict__ box_transform,
    Ray* __restrict__ rays,
    tcnn::default_rng_t rng
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ray_idx >= num_rays) {
        return;
    } 

    // spread rays uniformly between images
    int image_id = floor((float)ray_idx * (float)num_images / (float)num_rays);

    const ImageMeta* meta = &images_meta[image_id];

    rng.advance(ray_idx * 2);
    float x = rng.next_float() * meta->width;
    float y = rng.next_float() * meta->height;

    const float* image = &images_data[meta->offset];

    undistort_point(meta, x, y);

    rays[ray_idx].origin = Vector3f(meta->transform.translation());
    rays[ray_idx].dir = Vector3f(x, y, 1.0).normalized();
    rays[ray_idx].color = linear_interpolaiton(x, y, image, meta);
    box_intersections(
        rays[ray_idx].origin,
        rays[ray_idx].dir,
        box_transform,
        meta->near,
        meta->far,
        rays[ray_idx].near,
        rays[ray_idx].far
    );
}


/**
 * @brief Samples points from rays uniformly with jitter
 * 
 * @param num_samples           number of points to sample (m). should be equal to `num_rays * num_samples_per_ray`
 * @param num_rays              number of rays to sample points on
 * @param num_samples_per_ray   number of samples per ray
 * @param rays                  rays to sample point on
 * @param[out] points           (m x 3) matrix to write sampled points to. rows are samples, cols are coordinates.
 *                              samples on the same ray are contiguous.
 * @param[out] dirs             (m x 3) matrix to write directions to. rows are samples, cols are vector components.
 */
__global__
void sample_points_uniform_kernel(
    int num_samples,
    int num_rays,
    int num_samples_per_ray,
    const Ray* __restrict__ rays,
    tcnn::MatrixView<float> points,
    tcnn::MatrixView<float> dirs,
    tcnn::default_rng_t rng
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_idx > num_samples) {
        return;
    }

    int ray_idx = sample_idx / num_samples_per_ray;
    int sample_idx_in_ray = sample_idx - ray_idx * num_samples_per_ray;

    const Ray* ray = &rays[ray_idx];

    float delta = ray->far - ray->near;
    float near = ray->near + delta * (float)sample_idx_in_ray / (float)num_samples_per_ray;
    float far = ray->near + delta * (float)(sample_idx_in_ray + 1) / (float)num_samples_per_ray;

    rng.advance(sample_idx);
    float t = near + rng.next_float() * (near - far);
    Vector3f p = ray->origin + t * ray->dir;

    points(sample_idx, 0) = p.x();
    points(sample_idx, 1) = p.y();
    points(sample_idx, 2) = p.z();

    dirs(sample_idx, 0) = ray->dir.x();
    dirs(sample_idx, 1) = ray->dir.y();
    dirs(sample_idx, 2) = ray->dir.z();
}

DataManager::DataManager(
    const std::shared_ptr<ImagesDataset> dataset, 
    Isometry3f box_transform, 
    float near,
    float far
) {
    dataset_ = dataset;
    near_ = near;
    far_ = far;

    box_transform_.resize(1);
    box_transform_.copy_from_host(&box_transform);
}

void DataManager::load_images() {
    load_images(dataset_->image_ids());
}

void DataManager::load_images(const std::vector<int>& image_ids) {
    std::vector<float> images_data;
    std::vector<ImageMeta> images_meta;

    int total_images_size = 0;
    for (int image_id : image_ids) {
        std::shared_ptr<Camera> camera = dataset_->get_image(image_id)->camera();
        total_images_size += camera->width * camera->height * NUM_CHANNELS;        
    }

    images_meta.reserve(image_ids.size());
    images_data.reserve(total_images_size);

    int offset = 0;

    for (int image_id : image_ids) {
        std::shared_ptr<ImageHandle> image_handle = dataset_->get_image(image_id);
        std::shared_ptr<Camera> camera = image_handle->camera();

        cv::Mat image = image_handle->load();
        image.convertTo(image, CV_32FC4, 1.0 / 255.0);

        ImageMeta image_meta{
            image.cols,                 // width
            image.rows,                 // height
            offset,                     // offset
            image_handle->transform(),  // transform
            { camera->fx, camera->fy }, // focal_dist
            { camera->cx, camera->cy }, // center_point
            {},                         // dist_coefs (copied later)
            near_                       // near
        };

        memset(&image_meta.dist_coefs, 0, sizeof(image_meta.dist_coefs));
        memcpy(&image_meta.dist_coefs, &camera->dist_coefs[0], camera->dist_coefs.size() * sizeof(float));

        images_meta.push_back(image_meta);
        offset += image.total() * image.elemSize();

        images_data.insert(images_data.end(), (float*)image.data, ((float*)image.data) + image.total());
    }

    images_meta_.resize_and_copy_from_host(images_meta);
    images_data_.resize_and_copy_from_host(images_data);
}

void DataManager::sample_rays(cudaStream_t stream, int num_rays) {
    CHECK_F(!images_meta_.size(), "Should call `DataManager::load_images()` first");

    if (!rays_.size()) {
        rays_.resize(num_rays);
    }
    
    rng_.advance();
    tcnn::linear_kernel(sample_rays_kernel, 0, stream, 
        num_rays,               // num_rays
        images_meta_.size(),    // num_images
        near_,                  // near
        far_,                   // far
        images_meta_.data(),    // images_meta
        images_data_.data(),    // images_data
        box_transform_.data(),  // box_transform
        rays_.data(),           // rays
        rng_                    // rng
    );
}

void DataManager::sample_points(
    cudaStream_t stream,
    int num_samples_per_ray,
    tcnn::GPUMatrix<float>& points, 
    tcnn::GPUMatrix<float>& dirs
) {
    int num_rays = rays_.size();
    int num_samples = num_rays * num_samples_per_ray;
    
    if (points.rows() != num_samples || points.cols() != 3) {
        points.resize(num_samples, 3);
    }

    if (dirs.rows() != num_samples || points.cols() != 3) {
        dirs.resize(num_samples, 3);
    }

    rng_.advance();
    tcnn::linear_kernel(sample_points_uniform_kernel, 0, stream, 
        num_samples,
        num_rays,
        num_samples_per_ray,
        rays_.data(),
        points.view(),
        dirs.view(),
        rng_
    );
}

void DataManager::free_gpu() {
    images_meta_.free_memory();
    images_data_.free_memory();
    rays_.free_memory();
}

}