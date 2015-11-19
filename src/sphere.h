#pragma once
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