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
#include "sphere.inl"


namespace Sphere{
	void printSphere(Sphere *_sphere){
		//TODO
	}
	
	Sphere* createSphere(glm::vec3 _pos, float _radius, int _matIdx){
		//allocate a Sphere struct
		Sphere *result = (Sphere*)malloc(sizeof(Sphere));
		//set the position and radius
		result->position = _pos;
		result->radius = _radius;
		//set the material index
		result->materialIdx = _matIdx;
		
		return result;
	}
	
}
