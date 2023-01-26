#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/filesystem.hpp>

#include "camera_model.h"

namespace nerf {

namespace fs = boost::filesystem;
using namespace Eigen;

class ImageHandle {
public:
    ImageHandle(int image_id, fs::path image_path);

    void set_camera(const std::shared_ptr<Camera> camera);
    void set_transform(const Vector3f& translation, const Quaternionf& rotation);
    
    cv::Mat load(bool cache=false);
    void clear_cache();
    
    Isometry3f transform() const { return transform_; }
    std::shared_ptr<Camera> camera() const { return camera_; }

private:
    int id_;
    fs::path path_;
    cv::Mat image_;
    std::shared_ptr<Camera> camera_;

    Isometry3f transform_;    
};

}