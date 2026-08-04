
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK ((size_t)64)

#define L (1ULL << 20) * (THREADS_PER_BLOCK/16)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define ITERS 800000

__global__ void vectorAdd1D(float *A, float *B) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (idx >= L) {
		return;
	}
	float a = A[idx];
	float b = B[idx];
	float c = 1.1;
	int i;

	for (i = 0; i < ITERS; i++) {
		a = fmaf(a, b, c); // a * b + c
		b = fmaf(a, b, c);
		c = fmaf(a, b, c);
		a = fmaf(a, b, c);
	}

	B[idx] = a;
}


int main() {
	size_t size1D;
	float *A_c, *B_c;
	cudaError_t err;


	size1D = L * sizeof(float);

	//initArray(A, L);

  cudaMalloc(&A_c, size1D); // error-checking missing
  cudaMalloc(&B_c, size1D); // error-checking missing
  
	int n_blocks = CEIL_DIV(L, THREADS_PER_BLOCK);
	vectorAdd1D<<<n_blocks, THREADS_PER_BLOCK>>>(A_c, B_c);
  
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 1D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	cudaFree(A_c);

	return 0;
}
