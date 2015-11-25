/*
*	Sphere class
*	Author: Justin Jensen
*	Date: 11/19/2015
*
*	Similar to the Mesh class, this class is only here so we can load it in and pass it straight to the GPU
*
*/

#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

class Sphere{
public:
	//position
	glm::vec4 position;
	//radius
	float radius;
	//material index
	int materialIdx;
	
	Sphere(float _rad, glm::vec4 _pos, int _matIdx){
		radius = _rad;
		position = glm::vec4(_pos);
		materialIdx = _matIdx;
	}
	
};