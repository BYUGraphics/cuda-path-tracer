/*
*	Scene namespace
*	Author(s): Justin Jensen, Jeremy Oborn
*	Date: 11/19/2015
*
*	This namespace loads a scene from a file and stores it in a Scene struct that makes it easy to pass all of the data to the GPU.
*
*/

#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mesh.h"
#include "sphere.h"
#include <string>
#include <vector>
#include "scene.inl"
#include "material.inl"

namespace Scene{
	void printSceneInfo(Scene *_scene){
		//print the number of meshes, spheres, materials, and textures
		printf("Scene info:\n");
		printf("Meshes: %d\n", _scene->numMeshes);
		printf("Spheres: %d\n", _scene->numSpheres);
		printf("Materials: %d\n", _scene->numMaterials);
		printf("Textures: %d\n", _scene->numTextures);
		printf("Camera:\nwidth: %d\nheight: %d\nsamples: %d\nfov: %lf\n", 
			   _scene->width, _scene->height, _scene->samples, _scene->fov);
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
		std::vector<Mesh::Mesh> meshes;
		std::vector<Sphere::Sphere> spheres;
		std::vector<Sphere::Sphere> lights;
		std::vector<Material::Material> materials;
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
				std::string width_str, height_str, samples_str, fov_str;
				ss >> width_str >> height_str >> samples_str >> fov_str;
				result->width = atoi(width_str.c_str());
				result->height = atoi(height_str.c_str());
				result->samples = atoi(samples_str.c_str());
				result->fov = atof(fov_str.c_str());
				result->max_depth = 4;
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
				Material::Material tmpmat;
				tmpmat.cdiff = diff;
				tmpmat.crefl = refl;
				tmpmat.crefr = refr;
				tmpmat.cemit = emit;
				tmpmat.diff = atof(sdiff.c_str());
				tmpmat.refl = atof(srefl.c_str());
				tmpmat.refr = atof(srefr.c_str());
				tmpmat.emit = atof(semit.c_str());
				tmpmat.ior = atof(sior.c_str());
				
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
				}
				materials.push_back(tmpmat);
			}
			if(tok=="o")
			{
				std::cout << "creating mesh" << std::endl;
				std::string objfile, smtl;
				ss >> objfile >> smtl;
				// TODO: create mesh
				// std::cout << "numMeshes " << &(result->numMeshes) << std::endl;
			}
			if(tok=="s")
			{
				std::cout << "creating sphere" << std::endl;
				std::string sradius, smtl;
				ss >> sradius >> smtl;
				glm::vec3 s_pos = read_vector(ss);
				Sphere::Sphere tmp;
				tmp.position = s_pos;
				tmp.radius = atof(sradius.c_str());
				tmp.materialIdx = atoi(smtl.c_str());
				spheres.push_back(tmp);
				// TODO: if material has emit>0, add to light list
				if(materials[tmp.materialIdx].emit>0)
				{
					lights.push_back(tmp);
				}
				
			}	
		}
		
		result->numMaterials = materials.size();
		result->numTextures = 0;
		result->numMeshes = 0;
		result->numSpheres = spheres.size();
		result->numLights = lights.size();

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
		
		size_t lightsMemSize = lights.size() * sizeof(Sphere::Sphere);
		result->lights = (Sphere::Sphere*)malloc(lightsMemSize);
		memcpy(result->lights, lights.data(), lightsMemSize);

		size_t materialsMemSize = materials.size() * sizeof(Material::Material);
		result->materials = (Material::Material*)malloc(materialsMemSize);
		memcpy(result->materials, materials.data(), materialsMemSize);
		
		
		//load the camera
		double fov = result->fov * (M_PI/180.0);//radians
		double aspect_ratio = (double)result->height/(double)result->width;

		//calculate image plane size in world space
		result->scene_width = tan(fov/2.f)*2.f;
		result->scene_height = tan((fov*aspect_ratio)/2.f)*2.f;
		result->pixel_width = result->scene_width/result->width;
		result->pixel_slice = result->pixel_width/result->samples;
		
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
			
			//we don't need to do anything fancy with the BVH since that is taken care of in buildBVH.inl
		}
		
		//allocate space for spheres
		Sphere::Sphere *d_spheres;
		cudaMalloc((void**) &d_spheres, _scene->numSpheres * sizeof(Sphere::Sphere));
		//copy the spheres to the device
		cudaMemcpy(d_spheres, _scene->spheres, _scene->numSpheres * sizeof(Sphere::Sphere), cudaMemcpyHostToDevice);
		//copy the d_spheres pointer to the d_scene spheres pointer
		cudaMemcpy(&(d_scene->spheres), &d_spheres, sizeof(Sphere::Sphere*), cudaMemcpyHostToDevice);
		
		//allocate space fo lights
		Sphere::Sphere *d_lights;
		cudaMalloc((void**) &d_lights, _scene->numLights * sizeof(Sphere::Sphere));
		cudaMemcpy(d_lights, _scene->lights, _scene->numLights * sizeof(Sphere::Sphere), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_scene->lights, &d_lights, sizeof(Sphere::Sphere*), cudaMemcpyHostToDevice);

		//allocate space for materials
		Material::Material *d_materials;
		cudaMalloc((void**) &d_materials, _scene->numMaterials * sizeof(Material::Material));
		cudaMemcpy(d_materials, _scene->materials, _scene->numMaterials * sizeof(Material::Material), cudaMemcpyHostToDevice);
		cudaMemcpy(&d_scene->materials, &d_materials, sizeof(Material::Material*), cudaMemcpyHostToDevice);
		
		//allocate space for textures
		//TODO
		//allocate space for the camera
		//TODO
		
		
		return d_scene;
	}
}


















