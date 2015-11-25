#pragma once
//mesh.inl
//11/24/15

//include glm
#include <glm/glm.hpp>
//include bvh nodes
#include "bvh.inl"

namespace Mesh{
	struct Face{
		int numVertices, numNormals, numUVs;
		//list of vertex indices
		int *verts;
		//list of normal indices
		int *normals;
		//list of texture coordinates
		int *uvs;
	};
	
	struct Mesh{
		int numVerts, numFaces, numUVs, numNormals;
		//vertices*
		glm::vec3 *vertices;
		//triangles*
		Face *faces;
		glm::vec2 *uvs;
		glm::vec3 *normals;
		//BVH data
		BVH::BVH bvh;
		
		int materialIdx;
	};
	
	__device__ void intersectTriangle(Mesh *_mesh, int _triIdx, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx){
		//do a regular-old ray-triangle intersection test
		_intrsctDist = -1.f;
	}
	
	__device__ void intersectAABB(glm::vec3 _min, glm::vec3 _max, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist){
		//do a regular-old ray-box intersection test
		_intrsctDist = -1.f;
	}
	
	//device function intersect mesh()
	__device__ void intersectMesh(Mesh *_mesh, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx){
		//For now, loop through all the primitives and return the closest intersection
		int i;	//shared memory?
		float minDist = -1.f, tmpDist;	//shared memory?
		glm::vec3 minNormal, tmpNormal;	//shared memory?
		glm::vec2 minTexCoord, tmpTexCoord;	//shared memory?
		int minMatIdx, tmpMatIdx;	//shared memory?
		
		for(i = 0; i < _mesh->numFaces; i++){
			//intersect the triangle
			intersectTriangle(_mesh, i, _rayPos, _rayDir, tmpDist, tmpNormal, tmpTexCoord, tmpMatIdx);
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

















