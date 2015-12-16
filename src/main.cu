#include <test.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>
#include <math.h>
#include "scene.h"
#include "pixelmap.h"
#include "trace.h"
#include "buildBVH.inl"

double when()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

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
	
	double start = when();

	//render the image
	Pixelmap::Pixelmap *h_ppm = Pixelmap::create_pixelmap(h_scene->width, h_scene->height);
	glm::vec3 *d_pixels = Pixelmap::copyPixelmapToDevice(h_ppm);
	int bucketSize = 16;
	double xBuckets = (double)h_scene->width/(double)bucketSize;
	double yBuckets = (double)h_scene->height/(double)bucketSize;
	dim3 blocks((int)ceil(xBuckets), (int)ceil(yBuckets));
	dim3 threads(bucketSize, bucketSize);
	trace_scene<<<blocks, threads>>>(d_scene, d_pixels);
	//TODO
	
	//copy the rendered image from the GPU to the host
	//TODO
	Pixelmap::copyPixelmapToHost(d_pixels, h_ppm);
	
	//save the image to disk
	//TODO
	Pixelmap::write_pixelmap("out.ppm", h_ppm);
	double time_gone_by = when() - start;
	printf("TOTAL TIME: %lf seconds\n", time_gone_by);
	
	return 0;
}


