
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK ((size_t)64)

#define NSTREAMS 16
#define L ((THREADS_PER_BLOCK * 31) * NSTREAMS)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

static void initArray(int *arr, size_t size) {
	size_t i;
	for (i = 0; i < size; i++) {
		arr[i] = rand();
	}
}

__global__ void vectorAdd1D(int *A, int *B, int *C, int n) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx >= n) {
		return;
	}

	C[idx] = A[idx] + B[idx];
}


int main() {
	int *A, *B, *C;
	size_t i, size1D, numElemsStream;

	int *A_c[NSTREAMS], *B_c[NSTREAMS], *C_c[NSTREAMS];
	cudaError_t err;


	size1D = L * sizeof(int);
	numElemsStream = L / NSTREAMS;
	cudaMallocHost(&A, size1D);
	cudaMallocHost(&B, size1D);
	cudaMallocHost(&C, size1D);

	initArray(A, L);
	initArray(B, L);

	cudaStream_t streams[NSTREAMS];
	
	for (i = 0; i < NSTREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}


	for (i = 0; i < NSTREAMS; i++) {
  	cudaMalloc(&A_c[i], size1D/NSTREAMS); // error-checking missing
  	cudaMalloc(&B_c[i], size1D/NSTREAMS);
  	cudaMalloc(&C_c[i], size1D/NSTREAMS);
	}
  
	for (i = 0; i < NSTREAMS; i++) {
  	cudaMemcpyAsync(A_c[i], A + i * numElemsStream, size1D/NSTREAMS, cudaMemcpyHostToDevice, streams[i]);
  	cudaMemcpyAsync(B_c[i], B + i * numElemsStream, size1D/NSTREAMS, cudaMemcpyHostToDevice, streams[i]);

		int n_blocks = CEIL_DIV(numElemsStream, THREADS_PER_BLOCK);
		vectorAdd1D<<<n_blocks, THREADS_PER_BLOCK, 0, streams[i]>>>(A_c[i], B_c[i], C_c[i], L/NSTREAMS);
  
		err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA error in 1D kernel: %s\n", cudaGetErrorString(err));
			exit(1);
		}

		cudaMemcpyAsync(C + i * numElemsStream, C_c[i], size1D/NSTREAMS, cudaMemcpyDeviceToHost, streams[i]);
	}

	for (i = 0; i < NSTREAMS; i++) {
		cudaStreamSynchronize(streams[i]); // wait for all streams to execute
	}
	
	for (i = 0; i < L; i++) {
		assert(C[i] == A[i] + B[i]);
	}

	for (i = 0; i < NSTREAMS; i++) {
		cudaFree(A_c[i]);
		cudaFree(B_c[i]);
		cudaFree(C_c[i]);
	}
	for (i = 0; i < NSTREAMS; i++) {
		cudaStreamDestroy(streams[i]);
	}
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C);

	return 0;
}
