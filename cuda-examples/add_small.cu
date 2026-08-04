
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK ((size_t)64)

#define L ((THREADS_PER_BLOCK * 4) + 1)
#define W ((THREADS_PER_BLOCK * 3) + 2)
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
	int *a1d, *b1d, *c1d;
	int (*a2d)[W], (*b2d)[W], (*c2d)[W];
	int (*a3d)[W][H], (*b3d)[W][H], (*c3d)[W][H];
	size_t i, j, k, size1D, size2D, size3D;

	int *a1d_c, *b1d_c, *c1d_c;
	int *a2d_c, *b2d_c, *c2d_c;
	int *a3d_c, *b3d_c, *c3d_c;
	cudaError_t err;


	size1D = L * sizeof(int);
	a1d = (int*)malloc(size1D);
	b1d = (int*)malloc(size1D);
	c1d = (int*)malloc(size1D);

	initArray(a1d, L);
	initArray(b1d, L);

  cudaMalloc(&a1d_c, size1D); // error-checking missing
  cudaMalloc(&b1d_c, size1D);
  cudaMalloc(&c1d_c, size1D);
  
  cudaMemcpy(a1d_c, a1d, size1D, cudaMemcpyHostToDevice);
  cudaMemcpy(b1d_c, b1d, size1D, cudaMemcpyHostToDevice);

	int n_blocks = CEIL_DIV(L, THREADS_PER_BLOCK);
	vectorAdd1D<<<n_blocks, THREADS_PER_BLOCK>>>(a1d_c, b1d_c, c1d_c);
  
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 1D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	cudaMemcpy(c1d, c1d_c, size1D, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < L; i++) {
		assert(c1d[i] == a1d[i] + b1d[i]);
	}

	cudaFree(a1d_c);
	cudaFree(b1d_c);
	cudaFree(c1d_c);
	free(a1d);
	free(b1d);
	free(c1d);

	size2D = L * sizeof(*a2d);

	a2d = (int(*)[W])malloc(size2D);
	b2d = (int(*)[W])malloc(size2D);
	c2d = (int(*)[W])malloc(size2D);

	initArray((int*)a2d, L * W);
	initArray((int*)b2d, L * W);
  cudaMalloc(&a2d_c, size2D);
  cudaMalloc(&b2d_c, size2D);
  cudaMalloc(&c2d_c, size2D);

  cudaMemcpy(a2d_c, a2d, size2D, cudaMemcpyHostToDevice);
  cudaMemcpy(b2d_c, b2d, size2D, cudaMemcpyHostToDevice);

	dim3 n_blocks2d;
	dim3 n_threads2d(16, 4);

	// c2d[y][x]; y -> row, x -> column

	n_blocks2d.x = CEIL_DIV(W, 16); // column
	n_blocks2d.y = CEIL_DIV(L, 4); // row

	vectorAdd2D<<<n_blocks2d, n_threads2d>>>(a2d_c, b2d_c, c2d_c);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 2D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	cudaMemcpy(c2d, c2d_c, size2D, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < L; i++) {
		for (j = 0; j < W; j++) {
			assert(c2d[i][j] == a2d[i][j] + b2d[i][j]);
		}
	}

	cudaFree(a2d_c);
	cudaFree(b2d_c);
	cudaFree(c2d_c);
	free(a2d);
	free(b2d);
	free(c2d);

	size3D = L * sizeof(*a3d);

	a3d = (int(*)[W][H])malloc(size3D);
	b3d = (int(*)[W][H])malloc(size3D);
	c3d = (int(*)[W][H])malloc(size3D);

	initArray((int*)a3d, L * W * H);
	initArray((int*)b3d, L * W * H);
  cudaMalloc(&a3d_c, size3D);
  cudaMalloc(&b3d_c, size3D);
  cudaMalloc(&c3d_c, size3D);

  cudaMemcpy(a3d_c, a3d, size3D, cudaMemcpyHostToDevice);
  cudaMemcpy(b3d_c, b3d, size3D, cudaMemcpyHostToDevice);

	dim3 n_blocks3d;
	dim3 n_threads3d(8, 4, 2);

	// c3d[z][y][x]; z -> outermost, x -> innermost

	n_blocks3d.x = CEIL_DIV(H, 8); // innermost
	n_blocks3d.y = CEIL_DIV(W, 4);
	n_blocks3d.z = CEIL_DIV(L, 2); // outermost

	vectorAdd3D<<<n_blocks3d, n_threads3d>>>(a3d_c, b3d_c, c3d_c);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 3D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	cudaMemcpy(c3d, c3d_c, size3D, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < L; i++) {
		for (j = 0; j < W; j++) {
			for (k = 0; k < H; k++) {
				assert(c3d[i][j][k] == a3d[i][j][k] + b3d[i][j][k]);
			}
		}
	}

	cudaFree(a3d_c);
	cudaFree(b3d_c);
	cudaFree(c3d_c);
	free(a3d);
	free(b3d);
	free(c3d);
	
	return 0;
}
