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
	
	bool loadScene(std::string _path){
		//load the materials
		//TODO
		
		//load the objects (meshes and spheres)
		//TODO
		
		//load the camera
		//TODO
		
		return true;	//success
	}
};
