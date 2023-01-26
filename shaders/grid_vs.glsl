#version 400

uniform mat4 inv_view_proj;

out vec3 near_point;
out vec3 far_point;

const vec3 vertices[4] = vec3[] (
    vec3(-1, -1,  0),
    vec3( 1, -1,  0),
    vec3(-1,  1,  0),
    vec3( 1,  1,  0)
);

const int indices[6] = int[] (
    0, 1, 3,
    0, 3, 2
);

vec3 unwrap(vec4 point) {
    vec4 point_ = inv_view_proj * point;
    return point_.xyz / point_.w;
}

void main() {
    vec3 p = vertices[indices[gl_VertexID]];

    near_point = unwrap(vec4(p.xy, -1, 1));
    far_point = unwrap(vec4(p.xy, 1, 1));

    gl_Position = vec4(p, 1.0);
}