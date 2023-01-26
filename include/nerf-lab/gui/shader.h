#include <string>
#include <map>

#include <GL/glew.h>
#include <Eigen/Core>

namespace nerf {

class Shader {
public:
    Shader(GLenum shader_type);
    const Shader& load_and_compile(std::string source_file);

private:
    GLuint shader_object_;
    GLenum shader_type_;

    friend class ShaderProgram;
};

class ShaderProgram {
public:
    ShaderProgram(GLenum mode, GLsizei count, GLint first=0);
    void add_shader(const Shader& shader);
    void add_shader(GLenum shader_type, std::string source_file);
    void link_and_validate();
    void use();
    void draw();
    void set_uniform(std::string name, Eigen::Matrix4f mat);
    void set_uniform(std::string name, float value);

private:
    GLint get_location(std::string name);

    GLuint program_object_;
    GLenum mode_;
    GLint first_;
    GLsizei count_;
    std::map<std::string, GLint> locations_;
};

}