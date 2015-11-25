#pragma once
//buildBVH.inl
//11/25/15
//this file contains the kernel for building the BVH on the GPU

#include "bvh.inl"

namespace BVH{
	__global__ void buildBVH(Mesh::Mesh _mesh, int _num){
		//This is where the BVH magic happens
		//TODO
	}
	
	void buildBVH(Scene::Scene *_h_scene, Scene::Scene *_d_scene){
		int i, numFaces;
		//for each mesh in the scene
		for(i = 0; i < _h_scene->numMeshes; i++){
			//find out how many primitives the mesh has (that will determine the number of threads to launch)
			numFaces = _h_scene->meshes[i].numFaces;
			buildBVH<<<1, numFaces>>>(_d_scene->meshes[i], numFaces);
			//copy the BVH data into the Mesh struct
			//TODO
		}
		
	}
	
	
}



