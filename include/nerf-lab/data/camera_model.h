#pragma once

#include <vector>

namespace nerf {

struct Camera {
    enum class ModelType {
        kSimplePinhole,
        kPinhole,
        kSimpleRadial,
        kRadial,
        kOpenCV,
        kFishEye,
        kFullOpenCV,

        kFirst = kSimplePinhole,
        kLast = kFullOpenCV,
    };

    Camera(int width, int height, float fx, float fy, float cx, float cy,
           std::vector<float> dist_coefs, ModelType model_type) 
    : width(width), height(height), fx(fx), fy(fy), cx(cx), cy(cy), 
        dist_coefs(dist_coefs), model_type(model_type) {}

    int width, height;
    float fx, fy;
    float cx, cy;
    std::vector<float> dist_coefs;
    ModelType model_type;
};

}