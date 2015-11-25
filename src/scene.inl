#pragma once
#include "mesh.inl"
//scene.inl
//11/24/15


namespace Scene{
	
	struct Scene{
		int numMeshes, numSpheres, numTextures, numMaterials;
		//meshes
		Mesh::Mesh *meshes;
		//spheres
		Sphere::Sphere *spheres;
		//materials
		//textures
		//camera
	};
	
	__device__ void intersectScene(Scene *_scene, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx){
		int i;	//shared memory?
		float minDist = -1.f, tmpDist;	//shared memory?
		glm::vec3 minNormal, tmpNormal;	//shared memory?
		glm::vec2 minTexCoord, tmpTexCoord;	//shared memory?
		int minMatIdx, tmpMatIdx;	//shared memory?
		
		//travese either the object or the acceleration structure to find the closest intersection
		//for each sphere
		for(i = 0; i < _scene->numSpheres; i++){
			//intersect the sphere
			Sphere::intersectSphere(&(_scene->spheres[i]), _rayPos, _rayDir, tmpDist, tmpNormal, tmpTexCoord, tmpMatIdx);
			//if the distance is >= 0 and less than the minimum distance
			if(tmpDist >= 0.f && tmpDist < minDist){
				//set the minimum distance to the new distance
				minDist = tmpDist;
				//set the normal, UV, and material to the new ones
				minNormal = tmpNormal;
				minTexCoord = tmpTexCoord;
				minMatIdx = tmpMatIdx;
			}
		}
			
		//for each mesh
		for(i = 0; i < _scene->numMeshes; i++){
			//intersect the sphere
			Mesh::intersectMesh(&(_scene->meshes[i]), _rayPos, _rayDir, tmpDist, tmpNormal, tmpTexCoord, tmpMatIdx);
			//if the distance is >= 0 and less than the minimum distance
			if(tmpDist >= 0.f && tmpDist < minDist){
				//set the minimum distance to the new distance
				minDist = tmpDist;
				//set the normal, UV, and material to the new ones
				minNormal = tmpNormal;
				minTexCoord = tmpTexCoord;
				minMatIdx = tmpMatIdx;
			}
		}
		
		//populate _intrsctDist, _intrsctNorm, _texCoor, and _matIdx with the results
		_intrsctDist = minDist;
		_intrsctNorm = minNormal;
		_texCoord = minTexCoord;
		_matIdx = minMatIdx;
	}
}