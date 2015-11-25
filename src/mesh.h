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
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "mesh.inl"


namespace Mesh{
	//forward declaration of functions
	Mesh* createMesh(std::string, int);
	bool loadOBJ(Mesh*, const char*);
	void triangulateMesh(Mesh*);
	void printVertices(Mesh*);
	void printFaces(Mesh*);
	void printNormals(Mesh*);
	void printUVcoords(Mesh*);
	void printMaterialIdx(Mesh*);
	
	void printVec4(glm::vec4 _in){printf("(%f, %f, %f, %f)\n", _in.x, _in.y, _in.z, _in.w);}
	void printVec3(glm::vec3 _in){printf("(%f, %f, %f)\n", _in.x, _in.y, _in.z);}
	void printVec2(glm::vec2 _in){printf("(%f, %f)\n", _in.x, _in.y);}
	
	Mesh* createMesh(std::string _filename, int _matIdx){
		//allocate a Mesh struct
		Mesh* result = (Mesh*)malloc(sizeof(Mesh));
		
		//load the OBJ file
		loadOBJ(result, _filename.c_str());
		
		//triangulate the mesh
		triangulateMesh(result);
		
		//set the material index
		result->materialIdx = _matIdx;
		
		return result;
	}
	
	//This function is a modified version of an OBJ loader borrowed from 
	//https://github.com/Tecla/Rayito/blob/master/Rayito_Stage7_QT/OBJMesh.cpp
	//Thanks Mike!
	bool loadOBJ(Mesh *_mesh, const char *_filename){
		std::ifstream input(_filename);
		if(!input.is_open()){
			printf("Failed to open %s\n", _filename);
			return false;
		}
		std::vector<glm::vec3> verts;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec2> uvs;
		std::vector<Face> faces;
		std::vector<int> f_vertexIndices;
		std::vector<int> f_normalIndices;
		std::vector<int> f_uvIndices;
		std::string lineStr;
		std::string command;
		while (input.good()){
			lineStr.clear();
			std::getline(input, lineStr);

			std::istringstream lineInput(lineStr);
			if (lineInput.eof()){
				continue;
			}

			command.clear();
			lineInput >> command;
			if (lineInput.fail()){
				continue;
			}

			if (command[0] == '#'){
				// Found a comment; eat it
			}
			else if (command == "v"){
				// NOTE: there is an optional w coordinate that we're ignoring here
				glm::vec3 v;
				lineInput >> v.x;
				lineInput >> v.y;
				lineInput >> v.z;
				verts.push_back(v);
			}
			else if (command == "vn"){
				glm::vec3 v;
				lineInput >> v.x;
				lineInput >> v.y;
				lineInput >> v.z;
				normals.push_back(v);
			}
			else if (command == "vt"){
				// Note: there's an optional w coordinate that we're ignoring here
				glm::vec2 uv;
				lineInput >> uv.x;
				lineInput >> uv.y;
				uvs.push_back(uv);
			}
			else if (command == "f"){
				f_vertexIndices.clear();
				f_normalIndices.clear();
				f_uvIndices.clear();
				while (lineInput.good()){
					int vi;
					lineInput >> vi;
					if (lineInput.fail())
						break;
					int uvi, ni;
					bool gotUV = false;
					bool gotN = false;
					if (lineInput.peek() == '/'){
						char slash;
						lineInput >> slash;
						if (lineInput.peek() == '/'){
							lineInput >> slash;
							lineInput >> ni;
							gotN = true;
						}
						else{
							lineInput >> uvi;
							gotUV = true;
							if (lineInput.peek() == '/'){
								lineInput >> slash;
								lineInput >> ni;
								gotN = true;
							}
						}
					}
					vi = vi > 0 ? vi - 1 : (int)verts.size() + vi;
					f_vertexIndices.push_back(vi);
					if (vi >= (int)verts.size())
						std::cerr << "Found out-of-range vertex index: " << vi << std::endl;
					if (gotUV){
						uvi = uvi > 0 ? uvi - 1 : (int)uvs.size() + uvi;
						f_uvIndices.push_back(uvi);
						if (uvi >= uvs.size())
							std::cerr << "Found out-of-range UV index: " << uvi << std::endl;
					}
					if (gotN){
						ni = ni > 0 ? ni - 1 : (int)normals.size() + ni;
						f_normalIndices.push_back(ni);
						if (ni >= (int)normals.size())
							std::cerr << "Found out-of-range N index: " << ni << std::endl;
					}
				}
				//copy the vectors into a Face struct
				Face tmpFace; //should this be a Face* pointing to a Face in the heap?
				//copy the vertex indices
				tmpFace.numVertices = f_vertexIndices.size();
				size_t vertMemSize = f_vertexIndices.size() * sizeof(int);
				tmpFace.verts = (int*)malloc(vertMemSize);
				memcpy(tmpFace.verts, f_vertexIndices.data(), vertMemSize);
				//copy the normal indices
				tmpFace.numNormals = f_normalIndices.size();
				size_t normalMemSize = f_normalIndices.size() * sizeof(int);
				tmpFace.normals = (int*)malloc(normalMemSize);
				memcpy(tmpFace.normals, f_normalIndices.data(), normalMemSize);
				//copy the texture coordinate indices
				tmpFace.numUVs = f_uvIndices.size();
				size_t uvMemSize = f_uvIndices.size() * sizeof(int);
				tmpFace.uvs = (int*)malloc(uvMemSize);
				memcpy(tmpFace.uvs, f_uvIndices.data(), uvMemSize);
				//add the Face struct to the list of faces
				faces.push_back(tmpFace);
			}
			/*else if (command == "usemtl"){
				//skip it
			}
			else if (command == "mtllib"){
				//skip it
			}
			else if (command == "s"){
				//skip it
			}*/
			else{
				//skip it
			}
		}
		//convert the vectors into arrays and put them into the Mesh arrays
		//vertices
		_mesh->numVerts = verts.size();
		size_t vertsMemSize = verts.size() * sizeof(glm::vec3);//figure out how much space
		_mesh->vertices = (glm::vec3*)malloc(vertsMemSize);//allocate space
		memcpy(_mesh->vertices, verts.data(), vertsMemSize);//copy the data into the array
		//normals
		_mesh->numNormals = normals.size();
		size_t normsMemSize = normals.size() * sizeof(glm::vec3);
		_mesh->normals = (glm::vec3*)malloc(normsMemSize);
		memcpy(_mesh->normals, normals.data(), normsMemSize);
		//faces
		_mesh->numFaces = faces.size();
		size_t facesMemSize = faces.size() * sizeof(Face);
		_mesh->faces = (Face*)malloc(facesMemSize);
		memcpy(_mesh->faces, faces.data(), facesMemSize);
		//uvs
		_mesh->numUVs = uvs.size();
		size_t uvsMemSize = uvs.size() * sizeof(glm::vec2);
		_mesh->uvs = (glm::vec2*)malloc(uvsMemSize);
		memcpy(_mesh->uvs, uvs.data(), uvsMemSize);
		
		return true;
	}
	
	void triangulateMesh(Mesh *_mesh){
		//find any faces with more than 3 vertices and subdivide those bad Larrys
		//TODO
	}
	
	//some functions for debuging
	void printVertices(Mesh *_mesh){
		int i;
		printf("\nVertices:\n");
		for(i = 0; i < _mesh->numVerts; i++){
			printVec3(_mesh->vertices[i]);
		}
	}
	void printFaces(Mesh *_mesh){
		int i, v;
		printf("\nFaces: (these indices are zero-based. In the OBJ file, they're one-based)\n");
		for(i = 0; i < _mesh->numFaces; i++){
			Face tmp = _mesh->faces[i];
			for(v = 0; v < tmp.numVertices; v++){
				//print the vertex index
				//print the uv coordinate index
				//print the normal index
				printf("%d", tmp.verts[v]);
				if(v < tmp.numUVs)
					printf("/%d", tmp.uvs[v]);
				if(v < tmp.numNormals)
					printf("/%d", tmp.normals[v]);
				printf("\t");
			}
			printf("\n");
		}
	}
	void printNormals(Mesh *_mesh){
		int i;
		printf("\nVertex Normals:\n");
		for(i = 0; i < _mesh->numNormals; i++){
			printVec3(_mesh->normals[i]);
		}
	}
	void printUVcoords(Mesh *_mesh){
		int i;
		printf("\nTexture Coordinates:\n");
		for(i = 0; i < _mesh->numUVs; i++){
			printVec2(_mesh->uvs[i]);
		}
	}
	void printMaterialIdx(Mesh *_mesh){
		printf("\nMaterial Index: %d\n", _mesh->materialIdx);
	}
	
	
}
