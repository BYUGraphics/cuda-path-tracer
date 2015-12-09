//material.inl
#pragma once
#include <glm/glm.hpp>

namespace Material
{
    struct Material
    {
        float diff;
        float refl;
        float refr;
        float emit;
        float ior;
        glm::vec3 cdiff;
        glm::vec3 crefl;
        glm::vec3 crefr;
        glm::vec3 cemit;
    };

    Material* create_material(float _diff, float _refl, float _refr, float _emit, float _ior, 
                              glm::vec3 _cdiff, glm::vec3 _crefl, glm::vec3 _crefr, glm::vec3 _cemit)
    {
        Material *result = (Material*)malloc(sizeof(Material));
        result->diff = _diff;
        result->refl = _refl;
        result->refr = _refr;
        result->emit = _emit;
        result->ior = _ior;
        result->cdiff = _cdiff;
        result->crefl = _crefl;
        result->crefr = _crefr;
        result->cemit = _cemit;
        return result;
    }

}