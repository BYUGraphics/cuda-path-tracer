#include <test.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>
#include "scene.h"
#include "buildBVH.inl"

int main(int argc, char *argv[]){
	
	printf("Hello, CUDA world!\n");
	
	//testGLM();
	
	//testCUDA();
	
	//testCUDAglm();
	
	//testMeshOBJLoad(std::string("scenes/tetrahedron.obj"));
	//testMeshOBJLoad(std::string("scenes/cube.obj"));
	//testMeshOBJLoad(std::string("scenes/cube_no_normals_no_uvs.obj"));
	//testMeshOBJLoad(std::string("scenes/armadillo.obj"));
	//testMeshOBJLoad(std::string("scenes/buddha.obj"));
	//testMeshOBJLoad(std::string("scenes/sibenik.obj"));
	//testMeshOBJLoad(std::string("scenes/bunny.obj"));
	
	//testSceneCreate();
	
	//testSceneCopyToDevice();
	
	if(argc < 2){
		printf("ERROR: please specify a scene to render: %s <scene.txt> [<output.png>]\n", argv[0]);
		return 0;
	}
	//get the path to the scene to render
	//get the name of the output image
	
	//load the scene
	Scene::Scene *h_scene = Scene::createScene(std::string(argv[1]));

	Scene::printSceneInfo(h_scene);
	
	//pass all of the scene data to the GPU
	Scene::Scene *d_scene = Scene::copySceneToDevice(h_scene);
	
	//have the device build the BVHs for all objects in the scene
	BVH::buildBVH(h_scene, d_scene);
	
	//render the image
	//TODO
	
	//copy the rendered image from the GPU to the host
	//TODO
	
	//save the image to disk
	//TODO
		
	
	return 0;
}


