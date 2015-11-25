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
	
}
