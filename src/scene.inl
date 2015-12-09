#pragma once
#include "mesh.inl"
#include "material.inl"
//scene.inl
//11/24/15


namespace Scene{
	
	struct Scene{
		int numMeshes, numSpheres, numTextures, numMaterials, numLights;
		//meshes
		Mesh::Mesh *meshes;
		//spheres
		Sphere::Sphere *spheres;
		//lights
		Sphere::Sphere *lights;
		//materials
		Material::Material *materials;
		//textures
		//camera
		int width, height, samples;
		float fov;
		float scene_width, scene_height, pixel_width, pixel_slice;
		int max_depth;
	};
	
	//This function assumes that _rayDir is normalized and that _intrsctNorm and _texCoord have been initialized
	//This function returns whether or not a scene object was intersected
	__device__ bool intersectScene(Scene *_scene, glm::vec3 _rayOrig, glm::vec3 _rayDir, float *_intrsctDist, glm::vec3 *_intrsctNorm, glm::vec2 *_texCoord, int *_matIdx){		
		int i;	//shared memory?
		float minDist = 1e30f, tmpDist;	//shared memory?
		glm::vec3 minNormal, tmpNormal;	//shared memory?
		glm::vec2 minTexCoord, tmpTexCoord;	//shared memory?
		int minMatIdx, tmpMatIdx;	//shared memory?
		bool didIntersect = false;
		bool tmpIntersected;
		//travese either the object or the acceleration structure to find the closest intersection
		//for each sphere
		for(i = 0; i < _scene->numSpheres; i++){
			//intersect the sphere
			tmpIntersected = Sphere::intersectSphere(&(_scene->spheres[i]), _rayOrig, _rayDir, &tmpDist, &tmpNormal, &tmpTexCoord, &tmpMatIdx);
			didIntersect |= tmpIntersected;
			//if the distance is >= 0 and less than the minimum distance
			if(tmpIntersected && tmpDist >= 0.f && tmpDist < minDist){
				//set the minimum distance to the new distance
				minDist = tmpDist;
				//set the normal, UV, and material to the new ones
				minNormal.x = tmpNormal.x;
				minNormal.y = tmpNormal.y;
				minNormal.z = tmpNormal.z;
				minTexCoord.x = tmpTexCoord.x;
				minTexCoord.y = tmpTexCoord.y;
				minMatIdx = tmpMatIdx;
			}
		}
		
		
		//for each mesh
		for(i = 0; i < _scene->numMeshes; i++){
			//intersect the sphere
			tmpIntersected = Mesh::intersectMesh(&(_scene->meshes[i]), _rayOrig, _rayDir, &tmpDist, &tmpNormal, &tmpTexCoord, &tmpMatIdx);
			didIntersect |= tmpIntersected;
			//if the distance is >= 0 and less than the minimum distance
			if(tmpIntersected && tmpDist >= 0.f && tmpDist < minDist){
				//set the minimum distance to the new distance
				minDist = tmpDist;
				//set the normal, UV, and material to the new ones
				minNormal.x = tmpNormal.x;
				minNormal.y = tmpNormal.y;
				minNormal.z = tmpNormal.z;
				minTexCoord.x = tmpTexCoord.x;
				minTexCoord.y = tmpTexCoord.y;
				minMatIdx = tmpMatIdx;
			}
		}
		
		
		//populate _intrsctDist, _intrsctNorm, _texCoor, and _matIdx with the results
		*(_intrsctDist) = minDist;
		_intrsctNorm->x = minNormal.x;
		_intrsctNorm->y = minNormal.y;
		_intrsctNorm->z = minNormal.z;
		_texCoord->x = minTexCoord.x;
		_texCoord->y = minTexCoord.y;
		*(_matIdx) = minMatIdx;
		return didIntersect;
	}
}