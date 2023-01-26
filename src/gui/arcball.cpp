#include <cmath>

#include "nerf-lab/gui/arcball.h"

namespace nerf
{

Arcball::Arcball(float width, float height, std::shared_ptr<MainCamera> camera, float rotation_sensitivity)
: origin_(Vector3f::Zero()),
  camera_(camera),
  window_width_(width),
  window_height_(height),
  rotation_sensitivity_(rotation_sensitivity),
  is_panning_(false),
  is_rotating_(false) {}

void Arcball::rotate(float dx, float dy) {
    Isometry3f& transform = camera_->transform();

    transform.translation() -= origin_;
    transform.affine() = (AngleAxisf(dx, camera_->up()) * AngleAxis(dy, camera_->right())).toRotationMatrix() * transform.affine();
    transform.translation() += origin_;
}

void Arcball::pan(float dx, float dy) {
    Vector3f translation = camera_->right() * dx + camera_->up() * dy;

    if (camera_->is_orthogonal()) {
        translation *= camera_->scale();
    } else {
        translation *= (camera_->position() - origin_).norm();
    }

    camera_->transform().translation() += translation;
    origin_ += translation;
}

void Arcball::zoom(float dz) {
    Isometry3f& transform = camera_->transform();

    transform.translation() -= origin_;
    transform.translation() *= 1.0 + dz;
    transform.translation() += origin_;
}

void Arcball::start_rotation() {
    is_rotating_ = true;
    prev_xpos_ = -1;
}

void Arcball::stop_rotation() {
    is_rotating_ = false;
}

void Arcball::start_panning() {
    is_panning_ = true;
    prev_xpos_ = -1;
}

void Arcball::stop_panning() {
    is_panning_ = false;
}

void Arcball::update(float xpos, float ypos) {
    if (!is_panning_ && !is_rotating_) {
        return;
    }

    if (prev_xpos_ == -1) {
        prev_xpos_ = xpos;
        prev_ypos_ = ypos;

        return;
    }

    float dx = xpos - prev_xpos_;
    float dy = ypos - prev_ypos_;

    if (is_panning_) {
        pan(-dx * 0.001, dy * 0.001);
    }

    if (is_rotating_) {
        dx = dx / window_width_ * 2 * M_PI * rotation_sensitivity_;
        dy = dy / window_height_ * M_PI * rotation_sensitivity_;

        rotate(dx, dy);
    }

    prev_xpos_ = xpos;
    prev_ypos_ = ypos;
}

void Arcball::update_window_size(float width, float height) {
    window_width_ = width;
    window_height_ = height;
}

void Arcball::reset() {
    origin_ = Vector3f::Zero();
    camera_->set_view(camera_->position(), origin_, Vector3f::UnitY());
}

} // namespace nerf
