#pragma once

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <loguru/loguru.hpp>

#include "nerf-lab/gui/main_camera.h"
#include "nerf-lab/gui/arcball.h"
#include "nerf-lab/gui/texture.h"
#include "nerf-lab/data/colmap_dataset.h"


namespace nerf {

using namespace Eigen;

class App {
public:
    App();
    ~App();

    void run();
    void gui();

    void open_project();

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

private:
    GLFWwindow* window_;
    int window_pos_x_, window_pos_y_;
    int window_height_, window_width_;

    std::shared_ptr<MainCamera> main_camera_;
    std::shared_ptr<Arcball> arcball_;
    std::shared_ptr<ImagesDataset> dataset_;

    int preview_index_;
    std::shared_ptr<Texture> camera_preview_tex_;
};

}