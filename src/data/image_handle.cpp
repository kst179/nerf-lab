#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <loguru/loguru.hpp>

#include "nerf-lab/data/image_handle.h"

namespace nerf {

ImageHandle::ImageHandle(int id, fs::path path) {
    id_ = id;
    path_ = path;
}

void ImageHandle::set_camera(const std::shared_ptr<Camera> camera) {
    camera_ = camera;
}

void ImageHandle::set_transform(const Vector3f& translation, const Quaternionf& rotation) {
    transform_ = Isometry3f::Identity();
    transform_.translation() = translation;
    transform_.matrix().block<3, 3>(0, 0) = rotation.toRotationMatrix();

    // std::stringstream t, r, m;
    // t << translation;
    // r << rotation;
    // m << transform_.matrix();
    // LOG_F(INFO, "image %s\nt: %s\nr: %s\nm: %s", path_.c_str(), t.str().c_str(), r.str().c_str(), m.str().c_str());
}

cv::Mat ImageHandle::load(bool cache) {
    if (!image_.empty()) {
        return image_;
    }

    cv::Mat image = cv::imread(path_.c_str(), cv::IMREAD_UNCHANGED);
    if (image.channels() == 3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGBA);
    }

    if (cache) {
        image_ = image;
    }

    return image;
}

void ImageHandle::clear_cache() {
    image_.release();
}

}