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
	void printSceneInfo(Scene *_scene){
		//print the number of meshes, spheres, materials, and textures
		printf("Scene info:\n");
		printf("Meshes: %d\n", _scene->numMeshes);
		printf("Spheres: %d\n", _scene->numSpheres);
		printf("Materials: %d\n", _scene->numMaterials);
		printf("Textures: %d\n", _scene->numTextures);
	}
	
	Scene* createScene(std::string _filename){
		//make a new Scene struct
		Scene *result = (Scene*)malloc(sizeof(Scene));
		//load the materials
		//TODO
		result->numMaterials = 0;
		result->numTextures = 0;
		
		std::vector<Mesh::Mesh> meshes;
		//for each mesh
			//make a mesh object
			//add it to this->meshes
		result->numMeshes = 0;
		
		//for each sphere
			//make a sphere object
			//add it to this->spheres
		std::vector<Sphere::Sphere> spheres;
		//for testing
		Sphere::Sphere tmp;
		tmp.position = glm::vec3();
		tmp.radius = 1.f;
		tmp.materialIdx = 0;
		spheres.push_back(tmp);
		//end testing
		result->numSpheres = spheres.size(); //find out how many there are
		size_t spheresMemSize = spheres.size() * sizeof(Sphere::Sphere); //figure out how much memory it will take
		result->spheres = (Sphere::Sphere*)malloc(spheresMemSize); //allocate space
		memcpy(result->spheres, spheres.data(), spheresMemSize); //copy the data into the array
		
		
		
		//load the camera
		//TODO
		
		return result;
	}
	
	//copies the scene to the device and returns a pointer to it
	Scene* copySceneToDevice(Scene* _scene){
		//create a pointer to a Scene struct destined to live on the device
		Scene *d_scene;
		//allocate space for the scene on the device
		cudaMalloc((void**) &d_scene, sizeof(Scene));
		//copy the scene to the device
		cudaMemcpy(d_scene, _scene, sizeof(Scene), cudaMemcpyHostToDevice);
		
		//allocate space for meshes
		Mesh::Mesh *d_meshes;
		cudaMalloc((void**) &d_meshes, _scene->numMeshes * sizeof(Mesh::Mesh));
		//copy the meshes to the device (doesn't copy the vertices, normals, faces, etc.)
		cudaMemcpy(d_meshes, _scene->meshes, _scene->numMeshes * sizeof(Mesh::Mesh), cudaMemcpyHostToDevice);
		//copy the d_meshes pointer to the d_scene meshes pointer
		cudaMemcpy(&(d_scene->meshes), &d_meshes, sizeof(Mesh::Mesh*), cudaMemcpyHostToDevice);
		//for each mesh
			//copy the vertices to the device
			//copy the faces to the device
			//copy the uvs to the device
			//copy the normals to the device
			//for each face
				//copy the vertex indices to the device
				//copy the normal indices to the device
				//copy the uv indices to the device
			//we don't need to do anything fancy with the BVH since the device will do all the allocation of that stuff
		
		//allocate space for spheres
		Sphere::Sphere *d_spheres;
		cudaMalloc((void**) &d_spheres, _scene->numSpheres * sizeof(Sphere::Sphere));
		//copy the spheres to the device
		cudaMemcpy(d_spheres, _scene->spheres, _scene->numSpheres * sizeof(Sphere::Sphere), cudaMemcpyHostToDevice);
		//copy the d_spheres pointer to the d_scene spheres pointer
		cudaMemcpy(&(d_scene->spheres), &d_spheres, sizeof(Sphere::Sphere*), cudaMemcpyHostToDevice);
		
		
		//allocate space for materials
		//TODO
		//allocate space for textures
		//TODO
		//allocate space for the camera
		//TODO
		
		//delete the host's copy of the scene?
		
		return d_scene;
	}
}


















