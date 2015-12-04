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

	glm::vec3 read_vector(std::stringstream& ss)
	{
		std::string x, y, z;
		ss >> x >> y >> z;
		return glm::vec3(atof(x.c_str()), atof(y.c_str()), atof(z.c_str()));
	}
	
	Scene* createScene(std::string _filename){
		//make a new Scene struct
		Scene *result = (Scene*)malloc(sizeof(Scene));
		// std::cout << result << std::endl;
		// std::cout << "size " << sizeof(Scene) << std::endl;
		//load the materials
		//TODO
		std::vector<Mesh::Mesh> meshes;
		std::vector<Sphere::Sphere> spheres;
		result->numMaterials = 0;
		result->numTextures = 0;
		result->numMeshes = 0;
		result->numSpheres = 0;
		result->numLights = 0;
		FILE *objfile = fopen(_filename.c_str(), "r");
		if(objfile==NULL)
		{
			std::cout << "Error loading file " << _filename << std::endl;
		}
		char line[256];
		while (fgets(line, sizeof(line), objfile))
		{
			std::stringstream ss;
			ss << line;
			std::string tok;
			ss >> tok;
			if(tok[0]=='#')
			{
				continue;
			}
			if(tok=="c")
			{
				std::cout << "creating camera" << std::endl;
				std::string width_str, height_str, fov_str, samples_str;
				ss >> width_str >> height_str >> fov_str >> samples_str;
				// TODO: create camera
			}
			if(tok=="m")
			{
				std::cout << "creating material" << std::endl;
				glm::vec3 diff = read_vector(ss);
				glm::vec3 refl = read_vector(ss);
				glm::vec3 refr = read_vector(ss);
				glm::vec3 emit = read_vector(ss);
				std::string sdiff, srefl, srefr, semit, sior;
				ss >> sdiff >> srefl >> srefr >> semit >> sior;
				// TODO: create material
				char peek = ss.peek();
				while(peek==' ')
				{
					ss.get();
					peek = ss.peek();
				}
				if(ss.peek()!='\n')
				{
					std::string map;
					ss >> map;
					// TODO: create pixelmap, add to material
					result->numTextures++;
				}
				result->numMaterials++;
			}
			if(tok=="o")
			{
				std::cout << "creating mesh" << std::endl;
				std::string objfile, smtl;
				ss >> objfile >> smtl;
				// TODO: create mesh
				// std::cout << "numMeshes " << &(result->numMeshes) << std::endl;
				// result->numMeshes++; // causes segfault, no idea why
			}
			if(tok=="s")
			{
				std::cout << "creating sphere" << std::endl;
				std::string sradius, smtl;
				ss >> sradius >> smtl;
				glm::vec3 s_pos = read_vector(ss);
				// TODO: create sphere
				Sphere::Sphere tmp;
				tmp.position = s_pos;
				tmp.radius = atof(sradius.c_str());
				tmp.materialIdx = atoi(smtl.c_str());
				spheres.push_back(tmp);
				result->numSpheres++;
				// TODO: if material has emit>0, add to light list
				
			}	
		}
		
		//for each mesh
			//make a mesh object
			//add it to this->meshes
		
		//for each sphere
			//make a sphere object
			//add it to this->spheres
		//for testing
		// Sphere::Sphere tmp;
		// tmp.position = glm::vec3();
		// tmp.radius = 1.f;
		// tmp.materialIdx = 0;
		// spheres.push_back(tmp);
		//end testing
		// result->numSpheres = spheres.size(); //find out how many there are
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
		//copy the d_meshes pointer to the d_scene.meshes pointer
		cudaMemcpy(&(d_scene->meshes), &d_meshes, sizeof(Mesh::Mesh*), cudaMemcpyHostToDevice);
		//for each mesh
		for(int m = 0, numMeshes = _scene->numMeshes; m < numMeshes; m++){
			Mesh::Mesh *curMesh = &(_scene->meshes[m]);
			//allocate space for vertices
			glm::vec3 *d_verts;
			cudaMalloc((void**) &d_verts, curMesh->numVerts * sizeof(glm::vec3));
			//copy the vertices to the device
			cudaMemcpy(d_verts, curMesh->vertices, curMesh->numVerts * sizeof(glm::vec3), cudaMemcpyHostToDevice);
			//copy the d_verts pointer to the d_scene->meshes[m].vertices pointer
			cudaMemcpy(&(d_scene->meshes[m].vertices), &d_verts, sizeof(glm::vec3*), cudaMemcpyHostToDevice);
			
			//allocate space for uvs
			glm::vec2 *d_uvs;
			cudaMalloc((void**) &d_uvs, curMesh->numUVs * sizeof(glm::vec2));
			//copy the uvs to the device
			cudaMemcpy(d_uvs, curMesh->uvs, curMesh->numUVs * sizeof(glm::vec2), cudaMemcpyHostToDevice);
			//copy the d_uvs pointer to the d_scene->meshes[m].uvs pointer
			cudaMemcpy(&(d_scene->meshes[m].uvs), &d_uvs, sizeof(glm::vec2*), cudaMemcpyHostToDevice);
			
			//allocate space for normals
			glm::vec3 *d_normals;
			cudaMalloc((void**) &d_normals, curMesh->numNormals * sizeof(glm::vec3));
			//copy the normals to the device
			cudaMemcpy(d_normals, curMesh->normals, curMesh->numNormals * sizeof(glm::vec3), cudaMemcpyHostToDevice);
			//copy the d_normals pointer to the d_scene->meshes[m].normals pointer
			cudaMemcpy(&(d_scene->meshes[m].normals), &d_normals, sizeof(glm::vec3*), cudaMemcpyHostToDevice);
			
			//allocate space for faces
			Mesh::Face *d_faces;
			cudaMalloc((void**) &d_faces, curMesh->numFaces * sizeof(Mesh::Face));
			//copy the faces to the device
			cudaMemcpy(d_faces, curMesh->faces, curMesh->numFaces * sizeof(Mesh::Face), cudaMemcpyHostToDevice);
			//copy the d_faces pointer to the d_scene->meshes[m].faces pointer
			cudaMemcpy(&(d_scene->meshes[m].faces), &d_faces, sizeof(Mesh::Face*), cudaMemcpyHostToDevice);
			//for each face
			for(int f = 0, numFaces = curMesh->numFaces; f < numFaces; f++){
				Mesh::Face *curFace = &(curMesh->faces[f]);
				//allocate space for vertex indices
				int *d_vertIdxs;
				cudaMalloc((void**) &d_vertIdxs, curFace->numVertices * sizeof(int));
				//copy the vertex indices to the device
				cudaMemcpy(d_vertIdxs, curFace->verts, curFace->numVertices * sizeof(int), cudaMemcpyHostToDevice);
				//copy the d_vertIdxs pointer to the d_scene->mesh[m].faces[f].verts pointer
				cudaMemcpy(&(d_scene->meshes[m].faces[f].verts), &d_vertIdxs, sizeof(int*), cudaMemcpyHostToDevice);
				
				//allocate space for normal indices
				int *d_normIdxs;
				cudaMalloc((void**) &d_normIdxs, curFace->numNormals * sizeof(int));
				//copy the normal indices to the device
				cudaMemcpy(d_normIdxs, curFace->normals, curFace->numNormals * sizeof(int), cudaMemcpyHostToDevice);
				//copy the d_normIdxs pointer to the d_scene->mesh[m].faces[f].normals pointer
				cudaMemcpy(&(d_scene->meshes[m].faces[f].normals), &d_normIdxs, sizeof(int*), cudaMemcpyHostToDevice);
				
				//allocate space for uv indices
				int *d_uvIdxs;
				cudaMalloc((void**) &d_uvIdxs, curFace->numUVs * sizeof(int));
				//copy the uv indices to the device
				cudaMemcpy(d_uvIdxs, curFace->uvs, curFace->numUVs * sizeof(int), cudaMemcpyHostToDevice);
				//copy the d_normIdxs pointer to the d_scene->mesh[m].faces[f].uvs pointer
				cudaMemcpy(&(d_scene->meshes[m].faces[f].uvs), &d_uvIdxs, sizeof(int*), cudaMemcpyHostToDevice);
			}
			
			//we don't need to do anything fancy with the BVH since the device will do all the allocation of that stuff
		}
		
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
		
		
		return d_scene;
	}
}


















