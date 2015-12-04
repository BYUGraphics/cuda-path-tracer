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
	
	//do a regular-old ray-triangle intersection test
	//algorithm thanks to: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
	//Finding texture coordinates thanks to: http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
	__device__ bool intersectTriangle(Mesh *_mesh, int _triIdx, glm::vec3 _rayOrig, glm::vec3 _rayDir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx){
		//maybe put the ray into the triangle's space
		glm::vec3 e1, e2;
		glm::vec3 P, Q, T;
		float det, inv_det, u, v;
		float epsilon = 0.000001f;
		Face *faces = _mesh->faces;
		glm::vec3 *verts = _mesh->vertices;
		
		glm::vec3 v0 = verts[faces[_triIdx].verts[0]];
		glm::vec3 v1 = verts[faces[_triIdx].verts[1]];
		glm::vec3 v2 = verts[faces[_triIdx].verts[2]];
		
		//Find vectors for the two edges sharing V0
		e1 = v1 - v0;
		e2 = v2 - v0;
		
		//Begin calculating the determinant - also used to calculate the u parameter
		P = glm::cross(_rayDir, e2);
		
		//if the determinant is near zero, the ray lies in the plane of the triangle
		det = glm::dot(e1, P);
		if(det > -epsilon && det < epsilon)
			return false;
		inv_det = 1.f / det;
		
		//calculate the distance from V0 to the ray origin
		T = _rayOrig - v0;
		
		//calculate the u parameter and test bound
		u = glm::dot(T, P) * inv_det;
		
		//if the intersection lies outside the triangle
		if(u < 0.f || u > 0.f)
			return false;
		
		//prepare to test the v parameter
		Q = glm::cross(T, e1);
		
		//Calculate the v parameter and test the bound
		v = glm::dot(_rayDir, Q) * inv_det;
		
		//The intersection lies outside of the triangle
		if(v < 0.f || u + v > 1.f)
			return false;
		
		
		//WE'VE STRUCK TRIANGLE!!!!!!
		
		//if it got here, it's definitely inside the triangle
		_intrsctDist = glm::dot(e2, Q) * inv_det;
		
		//compute smoothed normals
		//surface normal = (1-u-v) * n0  +  u * n1  +  v * n2
		glm::vec3 *normals = _mesh->normals;
		glm::vec3 n0 = normals[faces[_triIdx].normals[0]];
		glm::vec3 n1 = normals[faces[_triIdx].normals[1]];
		glm::vec3 n2 = normals[faces[_triIdx].normals[2]];
		glm::vec3 norm = glm::normalize((1.f-u-v) * n0 + u * n1 + v * n2);
		_intrsctNorm.x = norm.x;
		_intrsctNorm.y = norm.y;
		_intrsctNorm.z = norm.z;
		
		//texture coord  = (1-u-v) * t0  +  u * t1  +  v * t2
		glm::vec2 *uvs = _mesh->uvs;
		glm::vec2 t0 = uvs[faces[_triIdx].uvs[0]];
		glm::vec2 t1 = uvs[faces[_triIdx].uvs[1]];
		glm::vec2 t2 = uvs[faces[_triIdx].uvs[2]];
		glm::vec2 texCoord = (1.f-u-v) * t0 + u * t1 + v * t2;
		_texCoord.x = texCoord.x;
		_texCoord.y = texCoord.y;
		
		//material index
		_matIdx = _mesh->materialIdx;
		
		return true;
	}
	
	__device__ bool intersectAABB(glm::vec3 _min, glm::vec3 _max, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist){
		//do a regular-old ray-box intersection test
		_intrsctDist = -1.f;
		return false;
	}
	
	//device function intersect mesh()
	__device__ bool intersectMesh(Mesh *_mesh, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx){
		//For now, loop through all the primitives and return the closest intersection
		int i, numFaces;	//shared memory?
		float minDist = -1.f, tmpDist;	//shared memory?
		glm::vec3 minNormal, tmpNormal;	//shared memory?
		glm::vec2 minTexCoord, tmpTexCoord;	//shared memory?
		int minMatIdx, tmpMatIdx;	//shared memory?
		bool didIntersect = false;	//shared memory?
		bool tmpIntrsct;	//shared memory?
		
		for(i = 0, numFaces = _mesh->numFaces; i < numFaces; i++){
			//intersect the triangle
			tmpIntrsct = intersectTriangle(_mesh, i, _rayPos, _rayDir, tmpDist, tmpNormal, tmpTexCoord, tmpMatIdx);
			didIntersect |= tmpIntrsct;
			//if the distance is >= 0 and less than the minimum distance
			if(tmpIntrsct && tmpDist >= 0.f && tmpDist < minDist){
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
		
		return didIntersect;
	}
	
}

















