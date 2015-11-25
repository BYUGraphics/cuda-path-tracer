#pragma once
#include "mesh.inl"
//scene.inl
//11/24/15


namespace Scene{
	
	struct Scene{
		//meshes
		Mesh::Mesh *meshes;
		//spheres
		//Sphere::Sphere *spheres;
		//materials
		//textures
		//camera
	};
	
	__device__ void intersectScene(/*pointer to the scene*/ glm::vec3 _rayPos, glm::vec3 _rayDir, glm::vec3 *_intrsctPos, glm::vec3 *_intrsctNorm, glm::vec2 *_texCoord, int *_matIdx){
		//travese either the object or the acceleration structure to find the closest intersection
		//populate _intrsctPos, _intrsctNorm, _texCoor, and _matIdx with the results
		
		//TODO
	}
}