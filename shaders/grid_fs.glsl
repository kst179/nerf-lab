#version 400

uniform mat4 view;
uniform mat4 proj;
uniform float far;

in vec3 near_point;
in vec3 far_point;

out vec4 fragColor;

const float scale = 1;

vec4 grid(vec2 point, float scale) {
    point /= scale;
    vec2 derivative = fwidth(point);
    vec2 grid = abs(fract(point + 0.5) - 0.5) / derivative;
    float alpha = 1 - min(grid.x, grid.y);

    if (alpha <= 0) {
        return vec4(0, 0, 0, 0);
    }

    float minimum_z = min(derivative.y, 1);
    float minimum_x = min(derivative.x, 1);

    vec4 color = vec4(0.2, 0.2, 0.2, alpha);

    if (-minimum_x < point.x && point.x < minimum_x) {
        color.z = 1.0;
    }
    if (-minimum_z < point.y && point.y < minimum_z) {
        color.x = 1.0;
    }

    return color;
}

void main() {
    vec3 delta = far_point - near_point;
    float t = - near_point.y / delta.y;
    
    if (t < 0) {
        discard;
    }

    vec3 hit_point = near_point + t * delta; 

    vec4 proj_point = view * vec4(hit_point, 1.0);

    float depth = proj_point.z; // linear depth

    proj_point = proj * proj_point;
    gl_FragDepth = proj_point.z / proj_point.w; // normalized depth

    float tg_alpha = abs(delta.y / length(delta.xz));
    float fading = max(0, 1 - depth / far);
    fading = min(fading, tg_alpha * 10);

    vec4 color = grid(hit_point.xz, 1) + 0.5 * grid(hit_point.xz, 10);
    
    color.a *= fading;

    if (color.a <= 0.1) {
        discard;
    }

    fragColor = color;
}