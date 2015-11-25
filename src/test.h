#pragma once
#define GLM_FORCE_CUDA
#include <cuda.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mesh.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

double When(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}
/*
void printVec4(glm::vec4 _in){
	printf("(%f, %f, %f, %f)\n", _in.x, _in.y, _in.z, _in.w);
}

void printVec3(glm::vec3 _in){
	printf("(%f, %f, %f)\n", _in.x, _in.y, _in.z);
}

void printVec2(glm::vec2 _in){
	printf("(%f, %f)\n", _in.x, _in.y);
}

__device__ void d_printVec3(glm::vec3 _in){
	printf("(%f, %f, %f)\n", _in.x, _in.y, _in.z);
}

void testGLM(){
	//some testing to make sure glm works
	glm::vec4 Position = glm::vec4(glm::vec3(0.0), 1.0);
	printVec4(Position);
	glm::mat4 Model = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f));
	glm::vec4 Transformed = Model * Position;
	printVec4(Transformed);
}

__device__ int add(int a, int b){
	return a + b;
}

__global__ void d_testCUDA(int* _in, int _incrementBy){
	_in[threadIdx.x] = add(_in[threadIdx.x], _incrementBy);
}

__global__ void d_testCUDAstruct(glm::vec3 *_array, double *_dbl_array, glm::vec3 trans){
	//d_printVec3(_array[threadIdx.x]);
	//_array[threadIdx.x] += trans;
	//d_printVec3(_array[threadIdx.x]);
	//store the result in the double array
	int startIdx = threadIdx.x * 3;
	_dbl_array[startIdx + 0] = _array[threadIdx.x].x + trans.x;
	_dbl_array[startIdx + 1] = _array[threadIdx.x].y + trans.y;
	_dbl_array[startIdx + 2] = _array[threadIdx.x].z + trans.z;
}

void testCUDA(){
	int i;
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
	d_testCUDA<<<numBlocks, blockSize>>>(d_a, incrementBy);
	//copy the device memory back to the host
	cudaMemcpy(h_b, d_a, sizeof(int) * num, cudaMemcpyDeviceToHost);
	printf("New array:\n");
	for(i = 0; i < num; i++){
		printf("%d\n", h_b[i]);
	}
}

void testCUDAglm(){
	int i;
	//create an array of 3 vec3s
	std::vector<glm::vec3> someVecs;
	someVecs.push_back(glm::vec3(0, 0, 0));
	someVecs.push_back(glm::vec3(1, 2, 3));
	someVecs.push_back(glm::vec3(4, 5, 6));
	
	glm::vec3 *someTransformedVecs = (glm::vec3*)malloc(3 * sizeof(glm::vec3));
	//create a translation vec3 
	glm::vec3 trans = glm::vec3(5, 6, 7);
	//allocate space on the GPU for the array and the vector
	glm::vec3 *d_array;
	double *d_dbl_array;
	double *h_dbl_array = (double*)malloc(3 * 3 * sizeof(double));
	cudaMalloc((void**) &d_array, 3 * sizeof(glm::vec3));
	cudaMalloc((void**) &d_dbl_array, 3 * 3 * sizeof(double));
	//get an array of the contents of someVecs
	glm::vec3 *vec_array = someVecs.data();
	//pass them to CUDA
	cudaMemcpy(d_array, &vec_array, 3 * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	
	printf("\nOriginal array of vec3s:\n");
	for(i = 0; i < 3; i++){
		printVec3(vec_array[i]);
	}
	
	//in CUDA, transform the vec3s
	d_testCUDAstruct<<<1, 3>>>(d_array, d_dbl_array, trans);
	
	//pass the transformed vec3s back in the form of an array
	cudaMemcpy(h_dbl_array, d_dbl_array, 3*3*sizeof(double), cudaMemcpyDeviceToHost);
	
	printf("\nTransformed coordinates:\n");
	for(i = 0; i < 9; i++){
		//printVec3(someTransformedVecs[i]);
		printf("%d: %f\n", i, h_dbl_array[i]);
	}
}

void testThrust(){
	//create an array of doubles on the host
	//copy it to the device
}*/

void testMeshOBJLoad(std::string _filename){
	//load tetrahedron.obj
	double start = When();
	Mesh::Mesh *shape = Mesh::createMesh(_filename, 0);
	double end = When();
	printf("%f seconds to load %s\n", end - start, _filename.c_str());
	//print its vertices
	//Mesh::printVertices(shape);
	//print its vertex normals
	//Mesh::printNormals(shape);
	//print its texture coordinates
	//Mesh::printUVcoords(shape);
	//print its faces
	//Mesh::printFaces(shape);
}