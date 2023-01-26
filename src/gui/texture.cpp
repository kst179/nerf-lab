#include <loguru/loguru.hpp>

#include "nerf-lab/gui/texture.h"

namespace nerf {

Texture::Texture(cv::Mat image) {
    CHECK_F(image.type() == CV_8UC3 || image.type() == CV_8UC4,
            "OGL Texture supports only 8-bit rgb or rgba images (3 or 4 channels), got image type %s", 
            cv::typeToString(image.type()).c_str());

    width_ = image.cols;
    height_ = image.rows;

    glGenTextures(1, &texture_object_);

    CHECK_F(texture_object_ != 0, "OGL: failed to create texture");

    GLenum target = GL_TEXTURE_2D;

    glBindTexture(target, texture_object_);

    glTexImage2D(target, 0, GL_RGBA, width_, height_, 0, 
                 image.channels() == 3 ? GL_RGB : GL_RGBA, GL_UNSIGNED_BYTE, image.data);

    glTexParameterf(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(target, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(target, GL_TEXTURE_WRAP_T, GL_CLAMP);

    glBindTexture(target, 0);
}

Texture::~Texture() {
    glDeleteTextures(1, &texture_object_);
}

void Texture::bind(GLenum texture_unit) {
    glActiveTexture(texture_unit);
    glBindTexture(GL_TEXTURE_2D, texture_object_);
}

GLuint Texture::texture_object() const { return texture_object_; }

}