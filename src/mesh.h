#pragma once
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>

class Mesh{
public:
	//path to .obj file
	//array of vertices
	//array of UV coordinates
	//array of vertex normals?
	//array of triangles
	//material index
	int materialIdx;
	Mesh(std::string _objPath, int _matIdx){
		//load the obj file
		//TODO
		
		materialIdx = _matIdx;
	}
};