#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <stdlib.h>
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
    // if(depth>=_scene->max_depth)
    // {
    //     return color;
    // }
    // float dist;
    // glm::vec3 norm(0.f);
    // glm::vec2 uv(0.f);
    // int matidx;

    glm::vec3 direction = dir;
    glm::vec3 origin = orig;

    for(int i=0; i<_scene->max_depth; i++)
    {
        // intersectScene(_scene, look_from, dir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx)
        // Hit::Hit hit = Scene::intersectScene(_scene, origin, direction, &dist, &norm, &uv, &matidx);
        Hit::Hit hit = Scene::intersectScene(_scene, origin, direction);
        if (hit.hit)
        {
            // calculate direct lighting for each light
            glm::vec3 hit_pt = origin + direction*hit.dist;
            for(int l=0; l<_scene->numLights; l++)
            {
                // glm::vec3 l(0,1,0);
                glm::vec3 light_dir = glm::normalize(_scene->lights[l].position - hit_pt);
                Material::Material light_material = _scene->materials[_scene->lights[l].materialIdx];
                float light_intensity = light_material.emit*0.2f;
                glm::vec3 light_color = light_material.cemit;

                // if(index==259680) printf("normal: %f, %f, %f\n", norm.x, norm.y, norm.z);
                glm::vec3 surface_color;
                if(_scene->materials[hit.matidx].diff>0.5)
                    surface_color = glm::vec3(1.f,0.f,0.f);//*glm::dot(norm, l)*3.f;
                else if(_scene->materials[hit.matidx].refl>0.5)
                    surface_color = glm::vec3(1.f,0.f,0.f);//*glm::dot(norm, l)*3.f;
                else if(_scene->materials[hit.matidx].refr>0.5)
                    surface_color = glm::vec3(1.f,0.f,0.f);//*glm::dot(norm, l)*3.f;

                // TODO: shadow ray

                float n_dot_l = glm::dot(hit.norm, light_dir);
                if(n_dot_l<0.f) n_dot_l = 0.f;
                color += surface_color * light_color * light_intensity * n_dot_l;
                
            }
            // reflection
            // glm::vec3 new_orig = orig + dir*dist + norm*(1e-30f);
            // glm::vec3 new_dir = norm*(glm::dot(norm, inc)*2.f) - inc;

            // printf("addpoint(geoself(), set(%.2f, %.2f, %.2f));\n", dir.x, dir.y, dir.z);
            // printf("addpoint(geoself(), set(%.2f, %.2f, %.2f));\n", orig.x, orig.y, orig.z);
            // printf("addpoint(geoself(), set(%.2f, %.2f, %.2f));\n", dist, 0.0f, 0.0f);

            // !===memory errors
            // printf("norm: %.2f, %.2f, %.2f\norig: %.2f, %.2f, %.2f\ndir: %.2f, %.2f, %.2f\ndist: %.2f\nhit_pt: %.2f, %.2f, %.2f\n\n", hit.norm.x, hit.norm.y, hit.norm.z, origin.x, origin.y, origin.z, direction.x, direction.y, direction.z, hit.dist, hit_pt.x, hit_pt.y, hit_pt.z);
            
            glm::vec3 old_orig = origin;
            glm::vec3 old_dir = direction;
            origin = hit_pt + hit.norm*(0.000000000000000001f);//1e-30f
            glm::vec3 inc = direction*(-1.f);
            glm::vec3 tmpdir = hit.norm*(glm::dot(hit.norm, inc)*2.f) - inc;
            tmpdir = glm::normalize(tmpdir);
            direction.x = tmpdir.x;
            direction.y = tmpdir.y;
            direction.z = tmpdir.z;

            // !===values totally messed up
            printf("new_orig: %.2f, %.2f, %.2f\nnorm: %.2f, %.2f, %.2f\norig: %.2f, %.2f, %.2f\ndir: %.2f, %.2f, %.2f\ndist: %.2f\nhit_pt: %.2f, %.2f, %.2f\n\n", origin.x, origin.y, origin.z, hit.norm.x, hit.norm.y, hit.norm.z, old_orig.x, old_orig.y, old_orig.z, old_dir.x, old_dir.y, old_dir.z, hit.dist, hit_pt.x, hit_pt.y, hit_pt.z);

            // printf("addpoint(geoself(), set(%.2f, %.2f, %.2f));\n", orig.x, orig.y, orig.z);
            // printf("addpoint(geoself(), set(%.2f, %.2f, %.2f));\n", hit_pt.x, hit_pt.y, hit_pt.z);

            // if(depth) printf("reflect vector: %f, %f, %f\n", dir.x, dir.y, dir.z);
        }
        else
        {
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
                glm::vec3 dir(xpos+(_scene->pixel_slice*i)+xoff, ypos+(_scene->pixel_slice*j)+yoff, depth);
                dir -= look_from;
                dir = glm::normalize(dir);
                color += trace_path(_scene, _pixels, dir, look_from, 0);
            }
        }
        color /= total_samples;
        _pixels[index] = clamp(color);
        
        

        // int ii = i - r.image_width/2;
        // int jj = j - r.image_height/2;
        // double x = ii*pixel_width + pixel_slice/2;
        // double y = jj*pixel_width + pixel_slice/2;
        
        // vec4 color(0,0,0);
        //for each sample
        // for(int m=0; m<r.samples1D; m++)
        // {
        //     for(int n=0; n<r.samples1D; n++)
        //     {
        //         double xoff = r.randd_negative()*pixel_slice;
        //         double yoff = r.randd_negative()*pixel_slice;
        //         vec4 dir(x+(pixel_slice*m)+xoff,y+(pixel_slice*n)+yoff,0);
        //         dir -= look_from;
        //         dir.normalize();
        //         ray v(look_from, dir);
        //         color += r.trace_path(v, 0, -1);
        //     }
        // }
        // color *= 1/samples;
        // color.clamp(1.0);
        // pic.setpixel(i, j, color);
        
    }
}