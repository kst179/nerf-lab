#include <stdexcept>
#include <fstream>
#include <sstream>

#include <loguru/loguru.hpp>

#include "nerf-lab/gui/shader.h"

namespace nerf {

std::string read_all(std::string path) {
    std::ifstream file(path);
    std::stringstream buf;
    buf << file.rdbuf();
    return buf.str();
}

Shader::Shader(GLenum shader_type) {
    shader_type_ = shader_type;
    shader_object_ = glCreateShader(shader_type);

    if (!shader_object_) {
        LOG_F(ERROR, "Cannot create shader object");
        throw std::runtime_error("Shader object cannot be created");
    }
}

const Shader& Shader::load_and_compile(std::string source_file) {
    std::string shader_source = read_all(source_file);
    const char* shader_source_c_str = shader_source.c_str();

    glShaderSource(shader_object_, 1, &shader_source_c_str, nullptr);
    glCompileShader(shader_object_);

    GLint success;
    glGetShaderiv(shader_object_, GL_COMPILE_STATUS, &success);
    
    if (!success) {
        GLchar compile_log[1024];
        glGetShaderInfoLog(shader_object_, 1024, nullptr, compile_log);

        ABORT_F("Shader %s filed to compile: %s", source_file.c_str(), compile_log);
    }

    return *this;
}

ShaderProgram::ShaderProgram(GLenum mode, GLsizei count, GLint first) {
    program_object_ = glCreateProgram();
    mode_ = mode;
    first_ = first;
    count_ = count;
}

void ShaderProgram::link_and_validate() {
    glLinkProgram(program_object_);

    GLint success;
    GLchar log[1024];
    glGetProgramiv(program_object_, GL_LINK_STATUS, &success);

    if (!success) {
        glGetProgramInfoLog(program_object_, sizeof(log), nullptr, log);
        LOG_F(ERROR, "program linking is failed: %s", log);
        throw std::runtime_error("program linking is failed");
    }

    glValidateProgram(program_object_);
    glGetProgramiv(program_object_, GL_VALIDATE_STATUS, &success);
    
    if (!success) {
        glGetProgramInfoLog(program_object_, sizeof(log), nullptr, log);
        LOG_F(ERROR, "program validation is failed: %s", log);
        throw std::runtime_error("program validation is failed");
    }
}

void ShaderProgram::use() {
    glUseProgram(program_object_);
}

void ShaderProgram::add_shader(const Shader& shader) {
    glAttachShader(program_object_, shader.shader_object_);
}

void ShaderProgram::add_shader(GLenum shader_type, std::string source_file) {
    Shader shader(shader_type);
    shader.load_and_compile(source_file);
    glAttachShader(program_object_, shader.shader_object_);
}

void ShaderProgram::set_uniform(std::string name, Eigen::Matrix4f mat) {
    glUniformMatrix4fv(get_location(name), 1, (GLboolean)mat.IsRowMajor, mat.data());
}

void ShaderProgram::set_uniform(std::string name, float value) {
    glUniform1f(get_location(name), value);
}

void ShaderProgram::draw() {
    glDrawArrays(mode_, first_, count_);
}

GLint ShaderProgram::get_location(std::string name) {
    auto it = locations_.find(name);
    GLint location;

    if (it == locations_.end()) {
        location = glGetUniformLocation(program_object_, (const GLchar*)name.c_str());
        locations_[name] = location;
    } else {
        location = it->second;
    }

    return location;
}

}