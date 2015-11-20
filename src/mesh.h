/*
*	Mesh class
*	Author(s): Justin Jensen
*	Date: 11/19/2015
*
*	The sole purpose of this class is to load a mesh from a file and store it in a way that makes it easy to pass all the data to CUDA.
*	We don't do anything fun with meshes on the CPU.
*
*/

#pragma once
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>



struct Face{
	std::vector<unsigned int> vertexIndices;
	std::vector<unsigned int> normalIndices;
	std::vector<unsigned int> uvIndices;
};


class Mesh{
public:
	//path to .obj file
	//array of vertices
	std::vector<glm::vec3> verts;	//these don't need to be vec4s because we know that they're all points. No need for homogenous coordinates yet.
	//array of UV coordinates
	std::vector<glm::vec2> uvs;
	//array of vertex normals
	std::vector<glm::vec3> normals;
	//array of faces
	std::vector<Face> faces;	//quads or triangles
	
	//material index
	int materialIdx;
	Mesh(std::string, int);
	
	bool loadOBJ(const char*);
	
	//some functions for debuging
	void printVertices();
	void printFaces(bool _showVerts);
	void printNormals();
	void printUVcoords();
	void printMaterialIdx();
	
private:
	//this function assumes all faces are convex
	void triangulate();
	void printVec4(glm::vec4 _in){printf("(%f, %f, %f, %f)\n", _in.x, _in.y, _in.z, _in.w);}
	void printVec3(glm::vec3 _in){printf("(%f, %f, %f)\n", _in.x, _in.y, _in.z);}
	void printVec2(glm::vec2 _in){printf("(%f, %f)\n", _in.x, _in.y);}
};