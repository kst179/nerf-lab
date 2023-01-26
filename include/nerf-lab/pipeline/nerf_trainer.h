#pragma once

#include <tiny-cuda-nn/object.h>

#include "nerf-lab/models/nerf.h"

namespace nerf {

class NerfTrainer : public tcnn::ObjectWithMutableHyperparams {
public:
    NerfTrainer() {}

private:
    std::shared_ptr<Nerf<float, __half>> model;
};

}