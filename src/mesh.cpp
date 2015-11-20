
#include <mesh.h>

Mesh::Mesh(std::string _objPath, int _matIdx){
	//load the obj file
	if(!this->loadOBJ(_objPath.c_str())){
		printf("ERROR: Failed to load %s\n", _objPath.c_str());
		verts.clear();
		uvs.clear();
		normals.clear();
		faces.clear();
	}
	
	this->triangulate();
	
	materialIdx = _matIdx;
}

//This function is a modified version of an OBJ loader borrowed from 
//https://github.com/Tecla/Rayito/blob/master/Rayito_Stage7_QT/OBJMesh.cpp
//Thanks Mike!
bool Mesh::loadOBJ(const char* _filename){
	std::ifstream input(_filename);
	if(!input.is_open()){
		printf("Failed to open %s\n", _filename);
		return false;
	}
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
			faces.push_back(Face());
			Face& face = faces.back();
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
				face.vertexIndices.push_back(vi);
				if (vi >= (int)verts.size())
					std::cerr << "Found out-of-range vertex index: " << vi << std::endl;
				if (gotUV){
                    uvi = uvi > 0 ? uvi - 1 : (int)uvs.size() + uvi;
                    face.uvIndices.push_back(uvi);
                    if (uvi >= uvs.size())
                        std::cerr << "Found out-of-range UV index: " << uvi << std::endl;
				}
				if (gotN){
					ni = ni > 0 ? ni - 1 : (int)normals.size() + ni;
					face.normalIndices.push_back(ni);
					if (ni >= (int)normals.size())
						std::cerr << "Found out-of-range N index: " << ni << std::endl;
				}
			}
		}
		else if (command == "usemtl"){
			//skip it
		}
		else if (command == "mtllib"){
			//skip it
		}
		else if (command == "s"){
			//skip it
		}
		else{
			//skip it
		}
	}
	return true;
}

void Mesh::printVertices(){
	int i, len;
	printf("\nVertices:\n");
	for(i = 0, len = verts.size(); i < len; i++){
		this->printVec3(verts[i]);
	}
}

void Mesh::printFaces(bool _showVerts){
	int i, v, len, vlen;
	printf("\nFaces: (these indices are zero-based. In the OBJ file, they're one-based)\n");
	for(i = 0, len = faces.size(); i < len; i++){
		Face tmp = faces[i];
		for(v = 0, vlen = tmp.vertexIndices.size(); v < vlen; v++){
			//print the vertex index
			//print the uv coordinate index
			//print the normal index
			printf("%d/%d/%d\t", tmp.vertexIndices[v], tmp.uvIndices[v], tmp.normalIndices[v]);
		}
		printf("\n");
	}
}

void Mesh::printNormals(){
	int i, len;
	printf("\nVertex Normals:\n");
	for(i = 0, len = normals.size(); i < len; i++){
		this->printVec3(normals[i]);
	}
}

void Mesh::printUVcoords(){
	int i, len;
	printf("\nTexture Coordinates:\n");
	for(i = 0, len = uvs.size(); i < len; i++){
		this->printVec2(uvs[i]);
	}
}

void Mesh::printMaterialIdx(){
	//TODO
}


//this function assumes that all faces are convex
void Mesh::triangulate(){
	//TODO
}


