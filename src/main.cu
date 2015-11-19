#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

__global__ void testCUDA(int* _in, int _incrementBy){
	_in[threadIdx.x] += _incrementBy;
}

void printVec4(glm::vec4 _in){
	printf("(%f, %f, %f, %f)\n", _in.x, _in.y, _in.z, _in.w);
}

int main(int argc, char *argv[]){
	int i;
	
	printf("Hello, CUDA world!\n");
	
	if(argc < 2){
		printf("ERROR: please specify a scene to render: %s <scene.txt> [<output.png>]\n", argv[0]);
	}
	//get the path to the scene to render
	//get the name of the output image
	
	//load the scene
	//TODO
	
	//render the image
	//TODO
	
	//save the image to disk
	//TODO
	
	//some testing to make sure glm works
	glm::vec4 Position = glm::vec4(glm::vec3(0.0), 1.0);
	printVec4(Position);
	glm::mat4 Model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f));
	glm::vec4 Transformed = Model * Position;
	printVec4(Transformed);
	
	//some testing to make sure CUDA works
	int numBlocks = 1;
	int num = 5;
	int blockSize = num;
	int incrementBy = 2;
	int *h_a, *h_b;	//pointers to host memory
	int *d_a;		//pointer to device memory
	h_a = (int*)malloc(num * sizeof(int));	//allocate the first array on the host
	h_b = (int*)malloc(num * sizeof(int));	//allocate the second array on the host
	cudaMalloc((void **) &d_a, num * sizeof(int));	//allocate memory on the device
	//initialize the host array a
	printf("Original array:\n");
	for(i = 0; i < num; i++){
		h_a[i] = i;
		printf("%d\n", h_a[i]);
	}
	//copy host array a to the device
	cudaMemcpy(d_a, h_a, sizeof(int) * num, cudaMemcpyHostToDevice);
	//run the CUDA kernel
	printf("Incrementing the array by %d\n", incrementBy);
	testCUDA<<<numBlocks, blockSize>>>(d_a, incrementBy);
	//copy the device memory back to the host
	cudaMemcpy(h_b, d_a, sizeof(int) * num, cudaMemcpyDeviceToHost);
	printf("New array:\n");
	for(i = 0; i < num; i++){
		printf("%d\n", h_b[i]);
	}
	
	
	return 0;
}


