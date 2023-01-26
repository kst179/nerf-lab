#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include "nerf-lab/gui/main_camera.h"

namespace nerf {

using namespace Eigen;

MainCamera::MainCamera(Vector3f position, Vector3f look_at, Vector3f up, 
                       float fov, float aspect, float near, float far) {
    set_view(position, look_at, up);
    
    fov_ = fov;
    aspect_ = aspect;
    near_ = near;
    far_ = far;

    scale_ = 1;

    arcball_origin_ = Vector3f::Zero();
}

Block<Matrix4f, 3, 1, true> MainCamera::position() { return transform_.translation(); }
Block<Matrix4f, 3, 1> MainCamera::right()    { return transform_.matrix().block<3, 1>(0, 0); }
Block<Matrix4f, 3, 1> MainCamera::up()       { return transform_.matrix().block<3, 1>(0, 1); }
Block<Matrix4f, 3, 1> MainCamera::forward()  { return transform_.matrix().block<3, 1>(0, 2); }

float MainCamera::near()   { return near_;   }
float MainCamera::far()    { return far_;    }
float MainCamera::aspect() { return aspect_; }

void MainCamera::set_translation(const Vector3f translation) {
    position() = translation;
}

void MainCamera::translate(const Vector3f translation) {
    transform_.translate(translation);
}

void MainCamera::set_view(Vector3f position, Vector3f look_at, Vector3f up_vec) {
    set_translation(position);

    Vector3f fwd_vec = (look_at - position).normalized();
    Vector3f right_vec = up_vec.cross(fwd_vec).normalized();
    up_vec = fwd_vec.cross(right_vec).normalized();

    right() = right_vec;
    up() = up_vec;
    forward() = fwd_vec;

    // std::stringstream ss;
    // ss << transform_.matrix();
    // LOG_F(INFO, "Camera matrix:\n%s", ss.str().c_str());
}

void MainCamera::set_view(Matrix4f view) {
    transform_ = view.inverse();
}

void MainCamera::set_camera_matrix(Matrix4f camera_matrix) {
    transform_ = camera_matrix;
}

Isometry3f& MainCamera::transform() {
    return transform_;
}

Matrix4f MainCamera::projection() const {
    if (orthogonal_) {
        Matrix4f projection_matrix {
            {1 / scale_ / aspect_, 0, 0, 0},
            {0, 1 / scale_, 0, 0},
            {0, 0, 0.01f / scale_, 0},
            {0, 0, 0, 1},
        };

        return projection_matrix;
    }

    float far_near_range = far_ - near_;

    float a = (far_ + near_) / far_near_range;
    float b = -2 * far_ * near_ / far_near_range;

    float f = 1 / tan(fov_ * 90 / M_PI);

    Matrix4f projection_matrix {
        {f/aspect_, 0, 0, 0},
        {0, f, 0, 0},
        {0, 0, a, b},
        {0, 0, 1, 0},
    };

    return projection_matrix;
}

Matrix4f MainCamera::view() const {
    return transform_.inverse().matrix();
}

Matrix4f MainCamera::view_proj() const {
    return projection() * view();
}

Matrix4f MainCamera::camera_matrix() const {
    return transform_.matrix();
}

void MainCamera::rotate(const AngleAxisf rotation) {
    transform_.rotate(rotation);
}

void MainCamera::ortho_zoom(float delta) {
    scale_ *= 1.0 + delta;
}


void MainCamera::change_aspect(float new_aspect) {
    aspect_ = new_aspect;
}

}