#pragma once
//bvh.inl
//11/24/15

#include <glm/glm.hpp>

//Forward declaration of the Mesh::Mesh struct
namespace Mesh{
	struct Mesh;
	__device__ bool intersectTriangle(Mesh*, int, glm::vec3, glm::vec3, float*, glm::vec3*, glm::vec2*, int*);
}

namespace BVH{
	
	//this struct just helps us sort the morton codes and object ids at the same time
	struct MortonPlusFaceID{
		unsigned int mortonCode;
		int faceIdx;
	};
	
	struct BVHnode{
		bool isLeaf;
		//int left child
		//int right child
		int left, right;
		//whether each child is a leaf node
		bool leftIsLeaf, rightIsLeaf;
		//the index of the parent node (useful when calculating the bounding boxes)
		int parent;
		//a lock for accessing the node and calculating its bounding box
		//initially set to -1, but then set to the ID of the first node that gets the lock
		int nodeLock;
		//initially set to false, but set to true once min and max have been calculated
		bool boundsSet;
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
		//array of object IDs, sorted by Morton Code
		int *sortedObjectIDs;
		//Pointer to the mesh associated with this BVH
		Mesh::Mesh *mesh;
	};
	
	//do a regular-old ray-box intersection test
	//this function assumes that _rayDir is normalized
	//ray-box intersection test thanks to: http://people.csail.mit.edu/amy/papers/box-jgt.pdf
	__device__ bool intersectAABB(glm::vec3 _min, glm::vec3 _max, glm::vec3 _rayOrig, glm::vec3 _rayDir, float *_intrsct0, float *_intrsct1){
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
		*(_intrsct0) = t0;
		*(_intrsct1) = t1;
		return t0 <= t1;
	}
	
	//this function is only called if we know the ray intersects it; we're just checking whether the ray intersects the children
	__device__ bool intersectBVHnode(BVH *_bvh, int _nodeIdx, bool _isInternal, glm::vec3 _rayOrig, glm::vec3 _rayDir, float *_intrsctDist, glm::vec3 *_intrsctNorm, glm::vec2 *_texCoord, int *_matIdx){
		BVHnode *curNode = _isInternal? &(_bvh->internalNodes[_nodeIdx]) : &(_bvh->leafNodes[_nodeIdx]);
		float bestDist = *(_intrsctDist), tmpDist;
		bool didIntersect = false, tmpIntersected;
		glm::vec3 minNormal, tmpNormal;
		glm::vec2 minTexCoord, tmpTexCoord;
		int minMatIdx, tmpMatIdx;
		
		//if it's an internal node
		if(_isInternal){
			float left_t0, left_t1, right_t0, right_t1;
			BVHnode *leftNode = curNode->leftIsLeaf? &(_bvh->leafNodes[curNode->left]) : &(_bvh->internalNodes[curNode->left]);
			//call intersectBVHnode on the left child
			bool leftInt = intersectAABB(leftNode->min, leftNode->max, _rayOrig, _rayDir, &left_t0, &left_t1);
			//if it intersects the left child
			if(leftInt){
				float left_t = left_t0 < 0? left_t1 : left_t0;
				bool isInside = left_t0 < 0;
				//if the AABB is closer than the closest triangle
				if(left_t < bestDist || isInside){
					//do an actual intersection on the contents of the node
					tmpIntersected = intersectBVHnode(_bvh, curNode->left, !(curNode->leftIsLeaf), _rayOrig, _rayDir, &tmpDist, &tmpNormal, &tmpTexCoord, &tmpMatIdx);
					didIntersect |= tmpIntersected;
					//if the distance is closer than the current best
					if(tmpIntersected && tmpDist >= 0.f && tmpDist < bestDist){
						//update the distance, normal, texture coordinate, and material index
						bestDist = tmpDist;
						minNormal.x = tmpNormal.x;
						minNormal.y = tmpNormal.y;
						minNormal.z = tmpNormal.z;
						minTexCoord.x = tmpTexCoord.x;
						minTexCoord.y = tmpTexCoord.y;
						minMatIdx = tmpMatIdx;
					}
				}
			}
			
			BVHnode *rightNode = curNode->rightIsLeaf? &(_bvh->leafNodes[curNode->right]) : &(_bvh->internalNodes[curNode->right]);
			//call intersectBVHnode on the left child
			bool rightInt = intersectAABB(rightNode->min, rightNode->max, _rayOrig, _rayDir, &right_t0, &right_t1);
			//if it intersects the left child
			if(rightInt){
				float right_t = right_t0 < 0? right_t1 : right_t0;
				bool isInside = right_t0 < 0;
				//if the AABB is closer than the closest triangle
				if(right_t < bestDist || isInside){
					//do an actual intersection on the contents of the node
					tmpIntersected = intersectBVHnode(_bvh, curNode->right, !(curNode->rightIsLeaf), _rayOrig, _rayDir, &tmpDist, &tmpNormal, &tmpTexCoord, &tmpMatIdx);
					didIntersect |= tmpIntersected;
					//if the distance is closer than the current best
					if(tmpIntersected && tmpDist >= 0.f && tmpDist < bestDist){
						//update the distance, normal, texture coordinate, and material index
						bestDist = tmpDist;
						minNormal.x = tmpNormal.x;
						minNormal.y = tmpNormal.y;
						minNormal.z = tmpNormal.z;
						minTexCoord.x = tmpTexCoord.x;
						minTexCoord.y = tmpTexCoord.y;
						minMatIdx = tmpMatIdx;
					}
				}
			}
			//return left intersected || right intersected
		}
		//if it's a leaf node
		else{
			//do ray-triangle intersection on the leaf's triangle
			//(the leaf's left child points to an index in bvh.d_sortedObjectIDs. That entry contains the face index in _mesh->faces)
			bool triInt = Mesh::intersectTriangle(_bvh->mesh, _bvh->sortedObjectIDs[curNode->left], _rayOrig, _rayDir, &tmpDist, &tmpNormal, &tmpTexCoord, &tmpMatIdx);
			didIntersect |= triInt;
			//if the ray hits the triangle
			if(triInt){
				//if it's closer than the current best
				if(tmpDist < bestDist){
					//update the distance, normal, texture coordinate, and material index
					bestDist = tmpDist;
					minNormal.x = tmpNormal.x;
					minNormal.y = tmpNormal.y;
					minNormal.z = tmpNormal.z;
					minTexCoord.x = tmpTexCoord.x;
					minTexCoord.y = tmpTexCoord.y;
					minMatIdx = tmpMatIdx;
				}
			}
		}
		
		//populate _intrsctDist, _intrsctNorm, _texCoor, and _matIdx with the results
		*(_intrsctDist) = bestDist;
		_intrsctNorm->x = minNormal.x;
		_intrsctNorm->y = minNormal.y;
		_intrsctNorm->z = minNormal.z;
		_texCoord->x = minTexCoord.x;
		_texCoord->y = minTexCoord.y;
		*(_matIdx) = minMatIdx;
		return didIntersect;
	}
	
	//traverse the BVH until you've either found a primitive, or you don't intersect anything
	__device__ bool intersectBVH(BVH *_bvh, glm::vec3 _rayOrig, glm::vec3 _rayDir, float *_intrsctDist, glm::vec3 *_intrsctNorm, glm::vec2 *_texCoord, int *_matIdx){
		//intersect the root node
		float t0, t1;
		BVHnode *root = &(_bvh->internalNodes[0]);
		bool intersectsRoot = intersectAABB(root->min, root->max, _rayOrig, _rayDir, &t0, &t1);
		//if it intersects the root node's AABB
		if(intersectsRoot){
			//call intersectBVHnode on the root node
			return intersectBVHnode(_bvh, 0, true, _rayOrig, _rayDir, _intrsctDist, _intrsctNorm, _texCoord, _matIdx);
		}
		return false;
	}
	
	
	
}
























