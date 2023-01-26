#version 400

out vec4 fragColor;
in vec4 pos;

void main() {
    gl_FragDepth = pos.z / pos.w;
    fragColor = vec4(1.0, 1.0, 1.0, 1.0);
}