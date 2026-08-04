
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

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
	int (*A)[W], (*B)[W], (*C)[W];
	size_t i, j, size2D;

	int *A_c, *B_c, *C_c;
	cudaError_t err;

	size2D = L * sizeof(*A);

	A = (int(*)[W])malloc(size2D);
	B = (int(*)[W])malloc(size2D);
	C = (int(*)[W])malloc(size2D);

	initArray((int*)A, L * W);
	initArray((int*)B, L * W);
  cudaMalloc(&A_c, size2D);
  cudaMalloc(&B_c, size2D);
  cudaMalloc(&C_c, size2D);

  cudaMemcpy(A_c, A, size2D, cudaMemcpyHostToDevice);
  cudaMemcpy(B_c, B, size2D, cudaMemcpyHostToDevice);

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

	cudaMemcpy(C, C_c, size2D, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < L; i++) {
		for (j = 0; j < W; j++) {
			assert(C[i][j] == A[i][j] + B[i][j]);
		}
	}

	cudaFree(A_c);
	cudaFree(B_c);
	cudaFree(C_c);
	free(A);
	free(B);
	free(C);

	return 0;
}
