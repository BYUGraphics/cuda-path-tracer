#pragma once
//buildBVH.inl
//11/25/15
//this file contains the kernel for building the BVH on the GPU

#include "bvh.inl"
#include <algorithm>	//std::sort

namespace BVH{
	
	__device__ int findSplit(unsigned int *_sortedMortonCodes, int _first, int _last){
		//get the first morton code in the range
		unsigned int firstCode = _sortedMortonCodes[_first];
		//get the last morton code in the range
		unsigned int lastCode = _sortedMortonCodes[_last];
		
		//if the first and last code are the same
		if(firstCode == lastCode){
			//return the average of their two indices
			return (_first + _last) >> 1;
		}
		
		//count the leading zeros that the two morton codes have in common
		int commonPrefix = __clz(firstCode ^ lastCode);
		
		//find the highest object that shares more than commonPrefix bits with the first one:
		//set split to first
		int split = _first;
		//set step to last - first
		int step = _last - _first;
		
		do{
			//set step to (step + 1) >> 1
			step = (step + 1) >> 1;
			//set newSplit to split + step
			int newSplit = split + step;
			//if the new split is less than last
			if(newSplit < _last){
				//get the morton code for the split index
				unsigned int splitCode = _sortedMortonCodes[newSplit];
				//get the number of prefix bits that the split and the first one have in common
				int splitPrefix = __clz(firstCode ^ splitCode);
				//if splitprefix is greater than the common prefix
				if(splitPrefix > commonPrefix){
					//split = newSplit
					split = newSplit;
				}
			}
		}
		while(step > 1);
		
		return split;
	}
	
	__device__ bool BVHnodeReadyToCalcBounds(Mesh::Mesh *_mesh, int _idx){
		bool leftReady = false, rightReady = false;
		//LEFT
		//if the left child is a leaf node
		if(_mesh->bvh.internalNodes[_idx].leftIsLeaf){
			leftReady = true;
		}
		else{
			leftReady = _mesh->bvh.internalNodes[_mesh->bvh.internalNodes[_idx].left].boundsSet;
		}
		
		//RIGHT
		//if the right child is a leaf node
		if(_mesh->bvh.internalNodes[_idx].rightIsLeaf){
			rightReady = true;
		}
		else{
			rightReady = _mesh->bvh.internalNodes[_mesh->bvh.internalNodes[_idx].right].boundsSet;
		}
		
		//if the children are ready return true
		return leftReady && rightReady;
	}
	
	__device__ void BVHnodeCalcBounds(Mesh::Mesh *_mesh, int _idx){
		//get the bounds of each child
		glm::vec3 leftMin, leftMax, rightMin, rightMax, min, max;
		if(_mesh->bvh.internalNodes[_idx].leftIsLeaf){
			leftMin = _mesh->bvh.leafNodes[_mesh->bvh.internalNodes[_idx].left].min;
			leftMax = _mesh->bvh.leafNodes[_mesh->bvh.internalNodes[_idx].left].max;
		}
		else{
			leftMin = _mesh->bvh.internalNodes[_mesh->bvh.internalNodes[_idx].left].min;
			leftMax = _mesh->bvh.internalNodes[_mesh->bvh.internalNodes[_idx].left].max;
		}
		
		if(_mesh->bvh.internalNodes[_idx].rightIsLeaf){
			rightMin = _mesh->bvh.leafNodes[_mesh->bvh.internalNodes[_idx].right].min;
			rightMax = _mesh->bvh.leafNodes[_mesh->bvh.internalNodes[_idx].right].max;
		}
		else{
			rightMin = _mesh->bvh.internalNodes[_mesh->bvh.internalNodes[_idx].right].min;
			rightMax = _mesh->bvh.internalNodes[_mesh->bvh.internalNodes[_idx].right].max;
		}
		//combine them
		min.x = glm::min(min.x, leftMin.x);
		min.y = glm::min(min.y, leftMin.y);
		min.z = glm::min(min.z, leftMin.z);
		min.x = glm::min(min.x, rightMin.x);
		min.y = glm::min(min.y, rightMin.y);
		min.z = glm::min(min.z, rightMin.z);
		
		max.x = glm::max(max.x, leftMax.x);
		max.y = glm::max(max.y, leftMax.y);
		max.z = glm::max(max.z, leftMax.z);
		max.x = glm::max(max.x, rightMax.x);
		max.y = glm::max(max.y, rightMax.y);
		max.z = glm::max(max.z, rightMax.z);
		
		_mesh->bvh.internalNodes[_idx].min.x = min.x;
		_mesh->bvh.internalNodes[_idx].min.y = min.y;
		_mesh->bvh.internalNodes[_idx].min.z = min.z;
		
		_mesh->bvh.internalNodes[_idx].max.x = max.x;
		_mesh->bvh.internalNodes[_idx].max.y = max.y;
		_mesh->bvh.internalNodes[_idx].max.z = max.z;
	}
	
	//This is where the BVH magic happens
	__global__ void buildBVHhierarchy(Mesh::Mesh _mesh, unsigned int * _sortedMortonCodes, int *_sortedFaceIDs, int _numFaces){
		//assign your leaf node the sorted face ID
		int idx = threadIdx.x;
		int faceIdx = _sortedFaceIDs[idx];
		_mesh.bvh.leafNodes[idx].isLeaf = true;
		_mesh.bvh.leafNodes[idx].left = faceIdx;
		//calculate the bounds of the triangle
		Mesh::Face myFace = _mesh.faces[faceIdx];
		glm::vec3 min = glm::vec3(1e30f);
		glm::vec3 max = glm::vec3(-1e30f);
		glm::vec3 *verts = _mesh.vertices;
		int i;
		for(i = 0; i < 3; i++){
			glm::vec3 tmpVert = verts[myFace.verts[i]];
			//update the min and max bounds
			min.x = glm::min(min.x, tmpVert.x);
			min.y = glm::min(min.y, tmpVert.y);
			min.z = glm::min(min.z, tmpVert.z);
			
			max.x = glm::max(max.x, tmpVert.x);
			max.y = glm::max(max.y, tmpVert.y);
			max.z = glm::max(max.z, tmpVert.z);
		}
		//store the bounds in the BVH
		_mesh.bvh.leafNodes[idx].min.x = min.x;
		_mesh.bvh.leafNodes[idx].min.y = min.y;
		_mesh.bvh.leafNodes[idx].min.z = min.z;
		_mesh.bvh.leafNodes[idx].max.x = max.x;
		_mesh.bvh.leafNodes[idx].max.y = max.y;
		_mesh.bvh.leafNodes[idx].max.z = max.z;
		_mesh.bvh.leafNodes[idx].boundsSet = true;
		
		if(idx >= _numFaces - 1)return;	//we only need numFaces - 1 threads working to generate the internal nodes
		
		//***Construct internal nodes***
		//set the internal node's lock to -1
		_mesh.bvh.internalNodes[idx].nodeLock = -1;
		_mesh.bvh.internalNodes[idx].parent = -1;
		_mesh.bvh.internalNodes[idx].boundsSet = false;
		_mesh.bvh.internalNodes[idx].isLeaf = false;
		
		int myCode = _sortedMortonCodes[idx];
		//determine my node's range
		int dir = __clz(myCode ^ _sortedMortonCodes[idx + 1]) - __clz(myCode ^ _sortedMortonCodes[idx - 1]) >= 0? 1 : -1;
		
		int Smin = __clz(myCode ^ _sortedMortonCodes[idx - dir]);
		int lmax = 2;
		while(__clz(myCode ^ _sortedMortonCodes[idx + lmax * dir]) > Smin){
			lmax = lmax << 1;
		}
		int len = 0;
		int shift = 1;
		int t;
		for(t = lmax >> shift; t >= 1; shift = shift << 1, t = lmax >> shift){
			//shift = shift << 1;
			if(__clz(myCode ^ _sortedMortonCodes[idx + (len + t) * dir]) > Smin){
				len += t;
			}
		}
		int j = idx + len * dir;
		//first
		int first = idx < j? idx : j;
		//last
		int last = idx > j? idx : j;
		
		//determine my range's split
		int split = findSplit(_sortedMortonCodes, first, last);
		
		//if split == first
		if(split == first){
			//my left child is leafNodes[split]
			_mesh.bvh.internalNodes[idx].left = split;
			_mesh.bvh.internalNodes[idx].leftIsLeaf = true;
			_mesh.bvh.leafNodes[split].parent = idx;
		}
		else{
			//my left child is internalNodes[split]
			_mesh.bvh.internalNodes[idx].left = split;
			_mesh.bvh.internalNodes[idx].leftIsLeaf = false;
			_mesh.bvh.internalNodes[split].parent = idx;
		}
		
		//if split + 1 == last
		if(split + 1 == last){
			//my right child is leafNodes[split+1]
			_mesh.bvh.internalNodes[idx].right = split + 1;
			_mesh.bvh.internalNodes[idx].rightIsLeaf = true;
			_mesh.bvh.leafNodes[split+1].parent = idx;
		}
		else{
			//my right child is internalNodes[split+1]
			_mesh.bvh.internalNodes[idx].right = split + 1;
			_mesh.bvh.internalNodes[idx].rightIsLeaf = false;
			_mesh.bvh.internalNodes[split+1].parent = idx;
		}
		
		__syncthreads();	//important, but this only works if all threads are in the same block
		
		//calculate the bounding box for each internal node, working from the leaves down to the root
		int curNodeIdx = idx;
		bool dropOut = false;
		while(curNodeIdx != -1 && !dropOut){
			//if you don't get a lock on the parent
			if(atomicExch(&(_mesh.bvh.internalNodes[curNodeIdx].nodeLock), idx) != -1){
				dropOut = true;
			}
			else{
				//if this node's children aren't ready
				if(!BVHnodeReadyToCalcBounds(&_mesh, curNodeIdx)){
					//put the -1 back into the lock
					atomicExch(&(_mesh.bvh.internalNodes[curNodeIdx].nodeLock), -1);
					//set dropout to true
					dropOut = true;
				}
				else{
					//calculate the bounding box of this node by combining its two children
					BVHnodeCalcBounds(&_mesh, curNodeIdx);
				}
			}
			//curNodeIdx = curNodeIdx's parent
			curNodeIdx = _mesh.bvh.internalNodes[curNodeIdx].parent;
			__syncthreads();
		}
	}
	
	__device__ unsigned int expandBits(unsigned int _v){
		_v = (_v * 0x00010001u) & 0xFF0000FFu;
		_v = (_v * 0x00000101u) & 0x0F00F00Fu;
		_v = (_v * 0x00000011u) & 0xC30C30C3u;
		_v = (_v * 0x00000005u) & 0x49249249u;
		return _v;
	}
	
	__device__ unsigned int morton3D(float _x, float _y, float _z){
		_x = min(max(_x * 1024.f, 0.f), 1023.f);
		_y = min(max(_y * 1024.f, 0.f), 1023.f);
		_z = min(max(_z * 1024.f, 0.f), 1023.f);
		unsigned int xx = expandBits((unsigned int)_x);
		unsigned int yy = expandBits((unsigned int)_y);
		unsigned int zz = expandBits((unsigned int)_z);
		return xx * 4 + yy * 2 + zz;
	}
	
	//generate the morton code for this face
	__global__ void generateMortonCodes(Mesh::Mesh *_mesh, MortonPlusFaceID *_mortonCodesPlusFaceIDs, int _numFaces){
		//get my thread ID
		//find out which face I'm responsible for
		int index = threadIdx.x;
		if(index >= _numFaces)return;
		
		Mesh::Face *myFace = &(_mesh->faces[index]);
		glm::vec3 midpoint;
		//calculate the midpoint of the face
		midpoint = (_mesh->vertices[myFace->verts[0]] +
					_mesh->vertices[myFace->verts[1]] +
					_mesh->vertices[myFace->verts[2]]) / 3.f;
		
		//calculate the morton code of the midpoint
		_mortonCodesPlusFaceIDs[index].mortonCode = morton3D(midpoint.x, midpoint.y, midpoint.z);
		_mortonCodesPlusFaceIDs[index].faceIdx = index;
	}
	
	bool compareMortonCodes(MortonPlusFaceID _a, MortonPlusFaceID _b){
		return _a.mortonCode < _b.mortonCode;
	}
	
	void buildBVH(Scene::Scene *_h_scene, Scene::Scene *_d_scene){
		int i, j, numFaces;
		//for each mesh in the scene
		for(i = 0; i < _h_scene->numMeshes; i++){
			//find out how many primitives the mesh has (that will determine the number of threads to launch)
			numFaces = _h_scene->meshes[i].numFaces;
			//allocate space for the BVHnode arrays
			//number of internal nodes = numFaces - 1
			BVHnode *d_internalNodes;
			cudaMalloc((void**) &d_internalNodes, (numFaces - 1) * sizeof(BVHnode));
			//number of leaf nodes = numFaces
			BVHnode *d_leafNodes;
			cudaMalloc((void**) &d_leafNodes, numFaces * sizeof(BVHnode));
			
			//put the internal node and leaf node arrays into the Mesh
			//TODO
			
			//allocate space for the morton codes
			MortonPlusFaceID *d_mortonCodesPlusFaceIDs;
			cudaMalloc((void**) &d_mortonCodesPlusFaceIDs, numFaces * sizeof(MortonPlusFaceID));
			//generate morton codes for each face (DEVICE)
			generateMortonCodes<<<1, numFaces>>>(&(_d_scene->meshes[i]), d_mortonCodesPlusFaceIDs, numFaces);
			//copy the morton codes to the host
			MortonPlusFaceID *h_mortonCodesPlusFaceIDs = (MortonPlusFaceID*)malloc(numFaces * sizeof(MortonPlusFaceID));
			cudaMemcpy(h_mortonCodesPlusFaceIDs, d_mortonCodesPlusFaceIDs, numFaces * sizeof(MortonPlusFaceID), cudaMemcpyDeviceToHost);
			
			//sort the faces by their morton code (HOST)
			std::sort(h_mortonCodesPlusFaceIDs, h_mortonCodesPlusFaceIDs + numFaces, compareMortonCodes);
			
			//create separate arrays of sorted morton codes and sorted face IDs
			unsigned int *h_sortedMortonCodes = (unsigned int*)malloc(numFaces * sizeof(unsigned int));
			int *h_sortedFaceIdxs = (int*)malloc(numFaces * sizeof(int));
			for(j = 0; j < numFaces; j++){
				h_sortedMortonCodes[j] = h_mortonCodesPlusFaceIDs[j].mortonCode;
				h_sortedFaceIdxs[j] = h_mortonCodesPlusFaceIDs[j].faceIdx;
			}
			//send those back to the device
			unsigned int *d_sortedMortonCodes;
			cudaMalloc((void**) &d_sortedMortonCodes, numFaces * sizeof(unsigned int));
			cudaMemcpy(d_sortedMortonCodes, h_sortedMortonCodes, numFaces * sizeof(unsigned int), cudaMemcpyHostToDevice);
			int *d_sortedFaceIdxs;
			cudaMalloc((void**) &d_sortedFaceIdxs, numFaces * sizeof(int));
			cudaMemcpy(d_sortedFaceIdxs, h_sortedFaceIdxs, numFaces * sizeof(int), cudaMemcpyHostToDevice);
			
			//build the BVH hiearchy
			buildBVHhierarchy<<<1, numFaces>>>(_d_scene->meshes[i], d_sortedMortonCodes, d_sortedFaceIdxs, numFaces);
			
			//calculate the bounding boxes for each BVH node (DEVICE)
			//TODO
			
			
			
			
			
			
			//cleanup:
			//device morton codes
			//host morton codes
		}
		
	}
	
	
}



