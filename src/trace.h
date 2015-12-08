


void __global__ trace_scene(Scene::Scene *_scene, glm::vec3 *_pixels)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x<_scene->width && y<_scene->height)
    {
        int xx = x - _scene->width/2;
        int yy = y - _scene->height/2;
        float xpos = xx*_scene->pixel_width;// + _scene->pixel_slice/2
        float ypos = yy*_scene->pixel_width;// + _scene->pixel_slice/2

        glm::vec3 color(0,0,0);

        glm::vec3 dir(xpos, ypos, 0);
        glm::vec3 look_from(0,0,1);
        dir -= look_from;
        dir = glm::normalize(dir);
        float dist;
        glm::vec3 norm;
        glm::vec2 uv;
        int matidx;
        // intersectScene(_scene, look_from, dir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx)
        bool hit = intersectScene(_scene, look_from, dir, dist, norm, uv, matidx);
        if (hit)
        {
            // int index = Pixelmap::index(_scene->width, x, y);
            int index = _scene->width*y + x;
            _pixels[index] = glm::vec3(1,1,1);
        }

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