#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <stdlib.h>
#include <stdio.h>
#include "scene.h"

//returns a random double between -0.5 and 0.5
float __device__ randd_negative()
{
    // return ((float)(rand() - RAND_MAX/2))/((float)RAND_MAX);
    // TODO: curand ???
    return 0.0;
}

glm::vec3 __device__ clamp(glm::vec3 vec)
{
    if(vec.x>1.f) vec.x=1.f;
    if(vec.y>1.f) vec.y=1.f;
    if(vec.z>1.f) vec.z=1.f;
    return vec;
}
// TODO: don't need _pixels
glm::vec3 __device__ trace_path(Scene::Scene *_scene, glm::vec3 *_pixels, glm::vec3 dir, glm::vec3 orig, int depth)
{
    glm::vec3 color(0.f,0.f,0.f);

    glm::vec3 direction = dir;
    glm::vec3 origin = orig;
    glm::vec3 ambient(0.1f);
    float mult = 1.0f;

    for(int i=0; i<_scene->max_depth; i++)
    {
        Hit::Hit hit = Scene::intersectScene(_scene, origin, direction);
        if (hit.hit)
        {
            glm::vec3 hit_pt = origin + direction*hit.dist;
            glm::vec3 inc;
            glm::vec3 tmpdir;

            // if we hit a light, return the emit color
            if(_scene->materials[hit.matidx].emit>0)
            {
                color += _scene->materials[hit.matidx].cemit*_scene->materials[hit.matidx].emit;
                break;
            }

            // calculate direct lighting for each light
            for(int l=0; l<_scene->numLights; l++)
            {
                // glm::vec3 l(0,1,0);
                glm::vec3 light_dir = glm::normalize(_scene->lights[l].position - hit_pt);
                Material::Material light_material = _scene->materials[_scene->lights[l].materialIdx];
                float light_intensity = light_material.emit;
                glm::vec3 light_color = light_material.cemit;

                glm::vec3 surface_color = _scene->materials[hit.matidx].cdiff*_scene->materials[hit.matidx].diff;
                float n_dot_l;

                // send shadow ray
                glm::vec3 light_orig = hit_pt + light_dir*0.00001f;
                Hit::Hit lightHit = Scene::intersectScene(_scene, light_orig, light_dir);
                // if we hit the light, shade the object
                if(_scene->materials[lightHit.matidx].emit>0)
                {
                    n_dot_l = glm::dot(hit.norm, light_dir);
                    if(n_dot_l<0.f) n_dot_l = 0.f;
                    color += surface_color * light_color * light_intensity * n_dot_l * mult + surface_color * ambient;//TODO: no ambient light
                }
                else
                {
                    color += surface_color * ambient;
                }
            }
            if(_scene->materials[hit.matidx].refl)
            {
                inc = direction*(-1.f);
                tmpdir = hit.norm*(glm::dot(hit.norm, inc)*2.f) - inc;
                tmpdir = glm::normalize(tmpdir);
                origin = hit_pt + tmpdir*(0.00001f);//1e-30f
                direction.x = tmpdir.x;
                direction.y = tmpdir.y;
                direction.z = tmpdir.z;
                mult = 0.5; //lose energy with each bounce
            }
            else if(_scene->materials[hit.matidx].refr)
            {
                //refraction
                bool into = glm::dot(hit.norm,direction)<=0;
                float n1 = (mult==1.f) ? 1.f : _scene->materials[hit.matidx].ior;
                float n2 = (!into) ? 1.f : _scene->materials[hit.matidx].ior;

                float R0s = ((n1 - n2)*(n1 - n2))/((n1+n2)*(n1+n2));
                float R0 = R0s*R0s;
                float R = R0 + (1.f-R0)*glm::pow((1.f-glm::dot(hit.norm, inc)), 5.f);
                float iof = n1/n2;
                float c1 = glm::dot(-hit.norm, direction);
                float cos2t = 1.f-iof*iof*(1.f-c1*c1);
                if (cos2t >= 0) 
                {
                    float c2 = glm::sqrt(cos2t);
                    direction = direction*iof + hit.norm * (iof*c1 - c2);
                    direction = glm::normalize(direction);
                    origin = hit_pt + direction*0.00001f;
                    mult = 0.5;
                }
                else // total internal reflection?
                {
                    break;
                }
            }
            else
            {
                break;
            }

            // if((blockIdx.x*blockDim.x + threadIdx.x)==553 && (blockIdx.y*blockDim.y + threadIdx.y)==241)
            //     printf("new_orig: %.2f, %.2f, %.2f\n new_dir: %.2f, %.2f, %.2f\nnorm: %.2f, %.2f, %.2f\norig: %.2f, %.2f, %.2f\ndir: %.2f, %.2f, %.2f\ndist: %.2f\nhit_pt: %.2f, %.2f, %.2f\n\n", origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, hit.norm.x, hit.norm.y, hit.norm.z, old_orig.x, old_orig.y, old_orig.z, old_dir.x, old_dir.y, old_dir.z, hit.dist, hit_pt.x, hit_pt.y, hit_pt.z);            

        }
        else
        {
            color += ambient;
            break;
        }
    }
    return color;
}

void __global__ trace_scene(Scene::Scene *_scene, glm::vec3 *_pixels)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<_scene->width && y<_scene->height)
    {
        int xx = x - _scene->width/2;
        int yy = y - _scene->height/2;
        float xpos = xx*_scene->pixel_width + _scene->pixel_slice/2;
        float ypos = yy*_scene->pixel_width + _scene->pixel_slice/2;
        float total_samples = _scene->samples * _scene->samples;
        int index = _scene->width*y + x;
        int depth = (index==259680) ? 1 : 0;

        glm::vec3 color(0,0,0);
        glm::vec3 look_from(0,0,1);
        for(int i=0; i<_scene->samples; i++)
        {
            for(int j=0; j<_scene->samples; j++)
            {
                double xoff = randd_negative()*_scene->pixel_slice;
                double yoff = randd_negative()*_scene->pixel_slice;
                glm::vec3 dir(xpos+(_scene->pixel_slice*i)+xoff, ypos+(_scene->pixel_slice*j)+yoff, 0);
                dir -= look_from;
                dir = glm::normalize(dir);
                color += trace_path(_scene, _pixels, dir, look_from, 0);
            }
        }
        color /= total_samples;
        _pixels[index] = clamp(color);
        
    }
}