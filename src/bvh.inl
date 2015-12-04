#pragma once
//bvhnodes.inl
//11/24/15

#include <glm/glm.hpp>

namespace BVH{
	
	struct BVHnode{
		//int left child
		//int right child
		int left, right;
		//vec3 min bounds
		glm::vec3 min;
		//vec3 max bounds
		glm::vec3 max;
	};

	struct BVH{
		//array of internal nodes
		BVHnode *internalNodes;
		//array of leaf nodes
		BVHnode *leafNodes;
	};
	
	//do a regular-old ray-box intersection test
	//this function assumes that _rayDir is normalized
	//ray-box intersection test thanks to: http://people.csail.mit.edu/amy/papers/box-jgt.pdf
	__device__ bool intersectAABB(glm::vec3 _min, glm::vec3 _max, glm::vec3 _rayOrig, glm::vec3 _rayDir, float &_intrsctDist){
		float t0 = -1e30f, t1 = 1e30f;
		glm::vec3 invDir = 1.f / _rayDir;
		glm::vec3 vt0 = (_min - _rayOrig) * invDir;
		glm::vec3 vt1 = (_max - _rayOrig) * invDir;
		glm::vec3 vtNear = glm::vec3(glm::min(vt0.x, vt1.x), glm::min(vt0.y, vt1.y), glm::min(vt0.z, vt1.z));
		glm::vec3 vtFar  = glm::vec3(glm::max(vt0.x, vt1.x), glm::max(vt0.y, vt1.y), glm::max(vt0.z, vt1.z));
		float btMin = glm::max(glm::max(vtNear.x, vtNear.y), max(vtNear.x, vtNear.z));
		float btMax = glm::min(glm::min(vtNear.x, vtNear.y), min(vtNear.x, vtNear.z));
		t0 = btMin > t0 ? btMin : t0;
		t1 = btMax < t1 ? btMax : t1;
		_intrsctDist = t0;
		return t0 <= t1;
	}
	
}
