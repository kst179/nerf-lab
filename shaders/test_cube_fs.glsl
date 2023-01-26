#version 400

in vec3 color;
in float depth;
out vec4 FragColor;

void main() {
    gl_FragDepth = depth;
    FragColor = vec4(color, 1.0);
}