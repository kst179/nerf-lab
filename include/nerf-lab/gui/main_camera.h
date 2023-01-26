#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <loguru/loguru.hpp>

namespace nerf {

using namespace Eigen;

class MainCamera {
public:
    MainCamera(Vector3f position, Vector3f look_at, Vector3f up, 
               float fov, float aspect, float near, float far);

    Matrix4f camera_matrix() const;
    Matrix4f view_proj() const;

    Isometry3f& transform();

    Block<Matrix4f, 3, 1, true> position();
    Block<Matrix4f, 3, 1> up();
    Block<Matrix4f, 3, 1> forward();
    Block<Matrix4f, 3, 1> right();
    float near();
    float far();
    float aspect();

    void set_translation(Vector3f translation);
    void translate(Vector3f translation);

    void set_view(Vector3f position, Vector3f look_at, Vector3f up);
    void set_view(Matrix4f view);
    void set_camera_matrix(Matrix4f camera_matrix);

    void set_perspective()  { orthogonal_ = false;        }
    void set_ortho()        { orthogonal_ = true;         }
    void toggle_ortho()     { orthogonal_ = !orthogonal_; }
    bool is_orthogonal() const { return orthogonal_; }
    float scale() const { return scale_; }


    void ortho_zoom(float delta);

    Matrix4f projection() const;
    Matrix4f view() const;

    void rotate(const AngleAxisf rotation);
    void arcball_rotate(float dx, float dy);
    void arcball_pan(float dx, float dy);
    void arcball_zoom(float delta);

    void change_aspect(float new_aspect);

private:
    Isometry3f transform_;
    float fov_;
    float aspect_;
    float near_;
    float far_;

    float scale_;
    bool orthogonal_;

    Vector3f arcball_origin_;
};

}