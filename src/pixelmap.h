#pragma once
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include "pixelmap.inl"

namespace Pixelmap
{

Pixelmap* create_pixelmap(int w, int h)
{
	Pixelmap *result = (Pixelmap*)malloc(sizeof(Pixelmap));
	result->width = w;
	result->height = h;
	result->total_size = w*h;
	result->pixels = (glm::vec3*)malloc(sizeof(glm::vec3)*result->total_size);
	for(int i=0; i<result->total_size; i++)
	{
		result->pixels[i] = glm::vec3(0,0,0);
	}
    return result;
}

int index(int width, int x, int y)
{
	return width*y + x;
}

void write_pixelmap(std::string _filename, Pixelmap *ppm)
{
	std::ofstream file;
  	file.open (_filename.c_str());
  	file << "P3\n" << ppm->width << " " << ppm->height << "\n" << 255 << "\n";

  	for(int i=ppm->height-1; i>=0; i--)
  	{
  		for(int j=0; j<ppm->width; j++)
  		{
			int idx = index(ppm->width, j, i);
  			file << round(ppm->pixels[idx].x*255) << " " 
  				 << round(ppm->pixels[idx].y*255) << " " 
  				 << round(ppm->pixels[idx].z*255) << "  ";
  		}
  		file << "\n";
  	}
  	file.close();
}

glm::vec3* copyPixelmapToDevice(Pixelmap *h_ppm)
{
    glm::vec3 *d_pixels;
    cudaMalloc((void**) &d_pixels, sizeof(glm::vec3)*h_ppm->total_size);    
    cudaMemcpy(d_pixels, h_ppm->pixels, sizeof(glm::vec3)*h_ppm->total_size, cudaMemcpyHostToDevice);
    return d_pixels;

    // Pixelmap *d_ppm;
    // //allocate space for the scene on the device
    // cudaMalloc((void**) &d_ppm, sizeof(Pixelmap));
    // //copy the scene to the device
    // cudaMemcpy(d_ppm, h_ppm, sizeof(Pixelmap), cudaMemcpyHostToDevice);


    // glm::vec3 *d_pixels;
    // cudaMalloc((void**) &d_pixels, h_ppm->total_size * sizeof(glm::vec3));
    // //copy the pixels to the device
    // cudaMemcpy(d_pixels, h_ppm->pixels, h_ppm->total_size * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    // //copy the d_meshes pointer to the d_scene.meshes pointer
    // cudaMemcpy(&(d_ppm->pixels), &d_pixels, sizeof(glm::vec3*), cudaMemcpyHostToDevice);
    // return d_ppm;
}

void copyPixelmapToHost(glm::vec3 *d_pixels, Pixelmap *h_ppm)
{
    cudaMemcpy(h_ppm->pixels, d_pixels, sizeof(glm::vec3)*h_ppm->total_size, cudaMemcpyDeviceToHost);
}


}
