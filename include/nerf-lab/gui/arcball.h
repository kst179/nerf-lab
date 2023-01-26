#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

#include "main_camera.h"

namespace nerf {

using namespace Eigen;

class Arcball{
public:
    Arcball(float width, float height, std::shared_ptr<MainCamera> camera, float rotation_sensitivity=2.0);

    void rotate(float dx, float dy);
    void pan(float dx, float dy);
    void zoom(float dz);
    void reset();

    void start_rotation();
    void stop_rotation();

    void start_panning();
    void stop_panning();

    void update(float dx, float dy);
    void update_window_size(float width, float height);

private:
    Vector3f origin_;
    std::shared_ptr<MainCamera> camera_;

    bool is_rotating_;
    bool is_panning_;

    float prev_xpos_;
    float prev_ypos_;

    float window_width_;
    float window_height_;

    float rotation_sensitivity_;
};

}