#pragma once

#include <GL/glew.h>
#include <opencv2/core.hpp>

namespace nerf {

class Texture {
public: 
    Texture(cv::Mat image);
    ~Texture();

    Texture(const Texture& other) = delete;
    Texture operator=(const Texture& other) = delete;

    int width() const { return width_; }
    int height() const { return height_; }

    void bind(GLenum texture_unit);

    GLuint texture_object() const;

private:
    GLuint texture_object_;

    int width_;
    int height_;
};

}