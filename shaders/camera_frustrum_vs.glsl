#version 400

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

out vec4 pos;

const vec3 vertices[6] = vec3[] (
    vec3(0, 0, 0),
    vec3(-1, -1, 1),
    vec3(1, -1, 1),
    vec3(-1, 1, 1),
    vec3(1, 1, 1),
    vec3(0, 1.5, 1)
);

const int indices[20] = int[] (
    0, 1,
    0, 2,
    0, 3,
    0, 4,
    
    1, 2,
    2, 4,
    4, 3,
    3, 1,

    4, 5,
    5, 3
);

void main() {
    vec4 p = vec4(vertices[indices[gl_VertexID]], 1.0);
    p.xyz *= 0.1;
    p = proj * view * model * p;
    pos = p;
    gl_Position = p;
}