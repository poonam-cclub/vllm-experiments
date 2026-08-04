
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK ((size_t)64)

#define L ((THREADS_PER_BLOCK * 31) + 1)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

static void initArray(int *arr, size_t size) {
	size_t i;
	for (i = 0; i < size; i++) {
		arr[i] = rand();
	}
}

__global__ void vectorAdd1D(int *A, int *B, int *C) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx >= L) {
		return;
	}

	C[idx] = A[idx] + B[idx];
}


int main() {
	int *A, *B, *C;
	size_t i, size1D;

	int *A_c, *B_c, *C_c;
	cudaError_t err;


	size1D = L * sizeof(int);
	A = (int*)malloc(size1D);
	B = (int*)malloc(size1D);
	C = (int*)malloc(size1D);

	initArray(A, L);
	initArray(B, L);

  cudaMalloc(&A_c, size1D); // error-checking missing
  cudaMalloc(&B_c, size1D);
  cudaMalloc(&C_c, size1D);
  
  cudaMemcpy(A_c, A, size1D, cudaMemcpyHostToDevice);
  cudaMemcpy(B_c, B, size1D, cudaMemcpyHostToDevice);

	int n_blocks = CEIL_DIV(L, THREADS_PER_BLOCK);
	vectorAdd1D<<<n_blocks, THREADS_PER_BLOCK>>>(A_c, B_c, C_c);
  
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 1D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	cudaMemcpy(C, C_c, size1D, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < L; i++) {
		assert(C[i] == A[i] + B[i]);
	}

	cudaFree(A_c);
	cudaFree(B_c);
	cudaFree(C_c);
	free(A);
	free(B);
	free(C);

	return 0;
}
