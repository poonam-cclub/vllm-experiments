
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>

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
	CUdevice device;
	int ua, managed;
	CUresult result;
	
	cuInit(0);
	cuDeviceGet(&device, 0);
	
	result = cuDeviceGetAttribute(&ua,  CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device);
	assert(result == CUDA_SUCCESS);
	if (ua == 0) {
		printf("Unified addressing not supported\n");
		exit(0);
	}

	
	result = cuDeviceGetAttribute(&managed,  CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, device);
	assert(result == CUDA_SUCCESS);
	if (managed == 0) {
		printf("managed memory not supported!\n");
		exit(0);
	}

	size_t i, size1D;

	int *A_c, *B_c, *C_c;
	cudaError_t err;

	size1D = L * sizeof(int);

  err = cudaMallocManaged(&A_c, size1D); // error-checking missing
	assert(err == cudaSuccess);
  err = cudaMallocManaged(&B_c, size1D);
	assert(err == cudaSuccess);
  err = cudaMallocManaged(&C_c, size1D);
	assert(err == cudaSuccess);


	initArray(A_c, L);
	initArray(B_c, L);

	int n_blocks = CEIL_DIV(L, THREADS_PER_BLOCK);
	vectorAdd1D<<<n_blocks, THREADS_PER_BLOCK>>>(A_c, B_c, C_c);
  
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 1D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	for (i = 0; i < L; i++) {
		assert(C_c[i] == A_c[i] + B_c[i]);
	}

	cudaFree(A_c);
	cudaFree(B_c);
	cudaFree(C_c);

	return 0;
}
