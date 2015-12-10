#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>

namespace Hit
{

    struct Hit{
        bool hit;
        float dist;
        glm::vec3 norm;
        glm::vec2 uv;
        int matidx;
    };
}