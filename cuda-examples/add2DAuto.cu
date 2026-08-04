
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define THREADS_PER_BLOCK ((size_t)64)

#define L ((THREADS_PER_BLOCK * 31) + 1)
#define W ((THREADS_PER_BLOCK * 15) + 2)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

static void initArray(int *arr, size_t size) {
	size_t i;
	for (i = 0; i < size; i++) {
		arr[i] = rand();
	}
}

__global__ void vectorAdd2D(int *A, int *B, int *C) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x; // column
	int y = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	if (x >= W || y >= L) {
		return;
	}

	int idx = (y * W) + x;  // A[y][x] --> y * W + x
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

	size_t i, j, size2D, idx;

	int *A_c, *B_c, *C_c;
	cudaError_t err;

	size2D = W * L * sizeof(int);


  err = cudaMallocManaged(&A_c, size2D);
	assert(err == cudaSuccess);
  err = cudaMallocManaged(&B_c, size2D);
	assert(err == cudaSuccess);
  err = cudaMallocManaged(&C_c, size2D);
	assert(err == cudaSuccess);
	
	initArray((int*)A_c, L * W);
	initArray((int*)B_c, L * W);

	dim3 n_blocks2d;
	dim3 n_threads2d(16, 4);

	// C[y][x]; y -> row, x -> column

	n_blocks2d.x = CEIL_DIV(W, 16); // column
	n_blocks2d.y = CEIL_DIV(L, 4); // row

	vectorAdd2D<<<n_blocks2d, n_threads2d>>>(A_c, B_c, C_c);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 2D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	for (i = 0; i < L; i++) {
		for (j = 0; j < W; j++) {
			idx = (i * W) + j;
			assert(C_c[idx] == A_c[idx] + B_c[idx]);
		}
	}

	cudaFree(A_c);
	cudaFree(B_c);
	cudaFree(C_c);

	return 0;
}
