#include <stdio.h>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/core.hpp>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <ImGuizmo/ImGuizmo.h>

#include "nerf-lab/gui/shader.h"
#include "nerf-lab/gui/texture.h"
#include "nerf-lab/gui/app.h"
#include "nerf-lab/data/colmap_dataset.h"

namespace nerf {

namespace fs = boost::filesystem;

App::App() {
    if (!glfwInit()) {
        LOG_F(ERROR, "GLFW init is failed");
        throw std::runtime_error("glfw init is failed");
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    window_ = glfwCreateWindow(mode->width, mode->height, "window", nullptr, nullptr);
    glfwSetWindowUserPointer(window_, this);

    if (!window_) {
        LOG_F(ERROR, "GLFW window cannot be created");
        glfwTerminate();
        throw std::runtime_error("glfw window cannot be created");
    }

    glfwMakeContextCurrent(window_);

    glewExperimental=GL_TRUE;
    GLenum glew_init_err = glewInit();

    if (GLEW_OK != glew_init_err) {
        LOG_F(ERROR, "GLEW init is failed");
        glfwDestroyWindow(window_);
        glfwTerminate();
        throw std::runtime_error("glew init is failed");
    }

    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    LOG_F(INFO, "Renderer: %s", renderer);
    LOG_F(INFO, "OpenGL version supported: %s", version);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // glEnable(GL_CULL_FACE);
    // glCullFace(GL_CW);

    glfwSetKeyCallback(window_, key_callback);
    glfwSetMouseButtonCallback(window_, mouse_button_callback);
    glfwSetCursorPosCallback(window_, cursor_position_callback);
    glfwSetScrollCallback(window_, scroll_callback);

    glfwSwapInterval(1);

    int width, height;
    glfwGetWindowSize(window_, &width, &height);

    main_camera_ = std::make_shared<MainCamera>(
        Eigen::Vector3f{2, 2, 2},       // position (from)
        Eigen::Vector3f{0, 0, 0},       // look_at  (to)
        Eigen::Vector3f::UnitY(),       // up
        60.0,                           // fov in degrees
        (float)width / (float)height,   // aspect
        0.1,                            // near
        100                             // far
    );
    arcball_ = std::make_shared<Arcball>(width, height, main_camera_);
}

void App::run() {
    ShaderProgram test_cube_shader(GL_TRIANGLES, 36);
    test_cube_shader.add_shader(GL_VERTEX_SHADER, "/home/kst179/fast-nerf/shaders/test_cube_vs.glsl");
    test_cube_shader.add_shader(GL_FRAGMENT_SHADER, "/home/kst179/fast-nerf/shaders/test_cube_fs.glsl");
    test_cube_shader.link_and_validate();

    ShaderProgram grid_shader(GL_TRIANGLES, 6);
    grid_shader.add_shader(GL_VERTEX_SHADER, "/home/kst179/fast-nerf/shaders/grid_vs.glsl");
    grid_shader.add_shader(GL_FRAGMENT_SHADER, "/home/kst179/fast-nerf/shaders/grid_fs.glsl");
    grid_shader.link_and_validate();

    ShaderProgram camera_frustrum(GL_LINES, 20);
    camera_frustrum.add_shader(GL_VERTEX_SHADER, "/home/kst179/fast-nerf/shaders/camera_frustrum_vs.glsl");
    camera_frustrum.add_shader(GL_FRAGMENT_SHADER, "/home/kst179/fast-nerf/shaders/camera_frustrum_fs.glsl");
    camera_frustrum.link_and_validate();

    /////

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init(nullptr);

    ImGui::GetStyle().ScaleAllSizes(2);
    ImGui::GetIO().FontGlobalScale = 2.5;
    ImGui::StyleColorsDark();

    /////

    while (!glfwWindowShouldClose(window_)) {
        int width, height;
        double time = glfwGetTime();

        glfwGetFramebufferSize(window_, &width, &height);
        glViewport(0, 0, width, height);
        main_camera_->change_aspect((float)width / (float)height);
        arcball_->update_window_size(width, height);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        // Matrix4f view = main_camera_->view();
        // ImGuizmo::ViewManipulate(view.data(), 0.1, ImVec2{(float)width - 200, 0}, ImVec2{200, 200}, 0);
        // main_camera_->set_view(view);

        // DRAW

        grid_shader.use();
        grid_shader.set_uniform("inv_view_proj", main_camera_->view_proj().inverse());
        grid_shader.set_uniform("view", main_camera_->view());
        grid_shader.set_uniform("proj", main_camera_->projection());
        grid_shader.set_uniform("far", main_camera_->far());
        grid_shader.draw();

        test_cube_shader.use();
        test_cube_shader.set_uniform("view_proj", main_camera_->view_proj());
        test_cube_shader.draw();

        // camera_frustrum.use();
        // camera_frustrum.set_uniform("view", main_camera_->view());
        // camera_frustrum.set_uniform("proj", main_camera_->projection());
        // camera_frustrum.set_uniform("model", Matrix4f::Identity());
        // camera_frustrum.draw();

        if (dataset_) {
            for (auto& [image_id, image] : dataset_->images()) {
                Isometry3f transform = image->transform();
                transform.matrix().block<3, 1>(0, 1) *= -1;

                camera_frustrum.use();
                camera_frustrum.set_uniform("view", main_camera_->view());
                camera_frustrum.set_uniform("proj", main_camera_->projection());
                camera_frustrum.set_uniform("model", transform.matrix());
                camera_frustrum.draw();
            }
        }

        // DRAW END
        // GUI

        gui();

        // ImGui::Image(,);

        // GUI END

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwPollEvents();
        glfwSwapBuffers(window_);
    }
}

void App::gui() {
    static bool preview;
    static int preview_item = 0;
    static std::vector<std::string> image_names;

    if (!dataset_) {
        dataset_ = std::make_shared<ColmapDataset>("/home/kst179/fast-nerf/data/lego");
        image_names = dataset_->image_names();
    }

    if (!dataset_) {
        ImGui::Begin("New project");
    } else {
        ImGui::Begin(dataset_->root().c_str());
    }
    
    if (ImGui::Button("Open project")) {
        open_project();

        if (dataset_) {
            image_names = dataset_->image_names();
        }
    }

    if (dataset_) {
        ImGui::Checkbox("preview cameras", &preview);
        if (preview) {
            int prev_item = preview_item;
            ImGui::ListBox("camera", &preview_item,
                [](void* data, int idx, const char **out_text) -> bool {
                    *out_text = ((std::string*)data)[idx].c_str();
                    return true;
                },
                &image_names[0],
                dataset_->size()
            );

            if (prev_item != preview_item) {
                camera_preview_tex_.reset();
            }
        }
    }

    ImGui::End();

    if (preview && dataset_) {
        if (!camera_preview_tex_) {
            camera_preview_tex_ = std::make_shared<Texture>(dataset_->load_image(preview_item));
        }

        ImGui::Begin("Camera preview");                                                                                                                                     

        ImGui::Image(
            (void*)(intptr_t)camera_preview_tex_->texture_object(), 
            ImVec2{ (float)camera_preview_tex_->width(), (float)camera_preview_tex_->height() }
        );

        ImGui::End();
    }
}

App::~App() {
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void App::open_project() {
    char root_c[PATH_MAX] = "";
    FILE* dialog = popen("zenity --file-selection --directory", "r");
    if (!fgets(root_c, PATH_MAX, dialog)) {
        LOG_F(INFO, "No dir selected");
    }

    std::string root(root_c);

    LOG_F(INFO, root.c_str());
    boost::trim(root);

    if (!root.empty()) {
        dataset_ = std::make_shared<ColmapDataset>(root);
    }
    pclose(dialog);
}

void App::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    App* app = (App*)glfwGetWindowUserPointer(window);

    // exit
    if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    // fullscreen toggle
    if (key == GLFW_KEY_F11 && action == GLFW_RELEASE) {
        GLFWmonitor* monitor = glfwGetWindowMonitor(window);
        if (!monitor) {      // windowed -> fullscreen mode
            glfwGetWindowPos(window, &app->window_pos_x_, &app->window_pos_y_);
            glfwGetWindowSize(window, &app->window_width_, &app->window_height_);

            monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(window, monitor, 0, 0, 
                                 mode->width, mode->height, mode->refreshRate);
        } else {            // fullscreen -> windowed mode
            glfwSetWindowMonitor(window, nullptr, 
                                 app->window_pos_x_, app->window_pos_y_, 
                                 app->window_width_, app->window_height_, 0);
        }
    }

    // arcball reset
    if (key == GLFW_KEY_C && action == GLFW_RELEASE) {
        app->arcball_->reset();
    }


    if (key == GLFW_KEY_0 && action == GLFW_RELEASE) {
        app->main_camera_->toggle_ortho();
    }
}

void App::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    App* app = (App*)glfwGetWindowUserPointer(window);
    
    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
        app->arcball_->start_rotation();
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_RELEASE) {
        app->arcball_->stop_rotation();
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        app->arcball_->start_panning();
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        app->arcball_->stop_panning();
    }
}

void App::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    App* app = (App*)glfwGetWindowUserPointer(window);

    if (!ImGui::GetIO().WantCaptureMouse) {
        app->arcball_->update(xpos, ypos);        
    }
}

void App::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    App* app = (App*)glfwGetWindowUserPointer(window);

    if (!ImGui::GetIO().WantCaptureMouse) {
        if (app->main_camera_->is_orthogonal()) {
            app->main_camera_->ortho_zoom(-yoffset * 0.1);
        } else {
            app->arcball_->zoom(-yoffset * 0.1);
        }
    }

}

};