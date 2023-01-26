#version 400

uniform mat4 view_proj;

const vec3 vertices[8] = vec3[] (
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 1.0, 0.0),

    vec3(0.0, 0.0, 1.0),
    vec3(1.0, 0.0, 1.0),
    vec3(1.0, 1.0, 1.0),
    vec3(0.0, 1.0, 1.0)
);

const int indices[36] = int[] (
    0, 1, 2,
    0, 2, 3,
    
    1, 5, 6,
    1, 6, 2,
    
    3, 2, 6,
    3, 6, 7,
    
    0, 4, 1,
    1, 4, 5,
    
    0, 3, 4,
    3, 7, 4,
    
    4, 7, 5,
    7, 6, 5
);

out vec3 color;
out float depth;

void main() {
    vec3 p = vertices[indices[gl_VertexID]] - 0.5;
    color = p;
    
    vec4 p_proj = view_proj * vec4(p, 1.0);
    depth = p_proj.z / p_proj.w;

    gl_Position = p_proj;
}
