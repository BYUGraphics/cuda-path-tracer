#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <scene.h>
#include <string>
#include <test.h>

int main(int argc, char *argv[]){
	
	printf("Hello, CUDA world!\n");
	
	testGLM();
	
	testCUDA();
	
	if(argc < 2){
		printf("ERROR: please specify a scene to render: %s <scene.txt> [<output.png>]\n", argv[0]);
		return 0;
	}
	//get the path to the scene to render
	//get the name of the output image
	
	//load the scene
	Scene *mainScene = new Scene();
	bool success = mainScene->loadScene(std::string(argv[1]));
	if(!success){
		printf("ERROR: Failed to load the scene\n");
		return 0;
	}
	
	//pass all of the scene data to the GPU...somehow
	//TODO
	
	//render the image
	//TODO
	
	//copy the rendered image from the GPU to the host
	
	//save the image to disk
	//TODO
		
	
	return 0;
}


