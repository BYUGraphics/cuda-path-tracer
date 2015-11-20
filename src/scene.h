/*
*	Scene class
*	Author(s): Justin Jensen
*	Date: 11/19/2015
*
*	This class loads a scene from a file and stores it in a way that makes it easy to pass all of the data to the GPU.
*
*/

#pragma once
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mesh.h>
#include <sphere.h>
#include <string>

class Scene{
public:
	//list of meshes
	Mesh* meshes;
	//list of spheres
	Sphere* spheres;
	//list of materials
	//TODO
	//camera
	//TODO
	
	Scene();
	~Scene();
	
	bool loadScene(std::string);
};
