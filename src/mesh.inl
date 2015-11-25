#pragma once
//mesh.inl
//11/24/15

//include glm
//include face
//include bvh nodes

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
		//BVHdata
		
		int materialIdx;
	};

	//device function intersect mesh()
	__device__ void intersectMesh(/*ray, out variables*/){
		//loop through all the primitives and return the closest intersection
	}
}