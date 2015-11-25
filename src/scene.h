/*
*	Scene namespace
*	Author(s): Justin Jensen
*	Date: 11/19/2015
*
*	This namespace loads a scene from a file and stores it in a Scene struct that makes it easy to pass all of the data to the GPU.
*
*/

#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mesh.h"
#include "sphere.h"
#include <string>
#include <vector>
#include "scene.inl"

namespace Scene{

	Scene* createScene(std::string _filename){
		//make a new Scene struct
		//load the materials
		//TODO
		
		//load the objects (meshes and spheres)
		//for each object
			//if it's a sphere
				//make a sphere object
				//add it to this->spheres
			//if it's a mesh
				//make a mesh object
				//add it to this->meshes
		
		//load the camera
		//TODO
		
		return NULL;
	}
}
