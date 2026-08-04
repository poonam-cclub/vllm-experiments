
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define THREADS_PER_BLOCK ((size_t)64)

#define L ((THREADS_PER_BLOCK * 31) + 1)
#define W ((THREADS_PER_BLOCK * 15) + 2)
#define H ((THREADS_PER_BLOCK * 2) + 3)

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

__global__ void vectorAdd2D(int *A, int *B, int *C) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x; // column
	int y = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	if (x >= W || y >= L) {
		return;
	}

	int idx = (y * W) + x;  // A[y][x] --> y * W + x
	C[idx] = A[idx] + B[idx];
}

__global__ void vectorAdd3D(int *A, int *B, int *C) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x; // innermost
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int z = (blockIdx.z * blockDim.z) + threadIdx.z; // outermost

	if (x >= H || y >= W || z >= L) {
		return;
	}

	int idx = (z * W * H) + (y * H) + x; // A[z][y][x] --> (z * W * H) + (y * H) + x
	
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

	size_t i, j, k, size3D, idx;

	int *A_c, *B_c, *C_c;
	cudaError_t err;

	size3D = L * W * H * sizeof(int);

  err = cudaMallocManaged(&A_c, size3D);
	assert(err == cudaSuccess);
  err = cudaMallocManaged(&B_c, size3D);
	assert(err == cudaSuccess);
  err = cudaMallocManaged(&C_c, size3D);
	assert(err == cudaSuccess);

	initArray((int*)A_c, L * W * H);
	initArray((int*)B_c, L * W * H);

	dim3 n_blocks3d;
	dim3 n_threads3d(8, 4, 2);

	// C[z][y][x]; z -> outermost, x -> innermost

	n_blocks3d.x = CEIL_DIV(H, 8); // innermost
	n_blocks3d.y = CEIL_DIV(W, 4);
	n_blocks3d.z = CEIL_DIV(L, 2); // outermost

	vectorAdd3D<<<n_blocks3d, n_threads3d>>>(A_c, B_c, C_c);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 3D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	for (i = 0; i < L; i++) {
		for (j = 0; j < W; j++) {
			for (k = 0; k < H; k++) {
				idx = (i * W * H) + (j * H) + k;
				assert(C_c[idx] == A_c[idx] + B_c[idx]);
			}
		}
	}

	cudaFree(A_c);
	cudaFree(B_c);
	cudaFree(C_c);

	return 0;
}
