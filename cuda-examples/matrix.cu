
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK ((size_t)64)
#define FACTOR 99

#define AROW ((THREADS_PER_BLOCK * FACTOR) + 1)
#define ACOL ((THREADS_PER_BLOCK * FACTOR) + 2)
#define BROW ACOL
#define BCOL ((THREADS_PER_BLOCK * FACTOR) + 3)
#define CROW AROW
#define CCOL BCOL

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

static void initArray(int *arr, size_t size) {
	size_t i;
	for (i = 0; i < size; i++) {
		arr[i] = rand();
	}
}


__global__ void matmul2D(int *A, int *B, int *C) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x; // column
	int y = (blockIdx.y * blockDim.y) + threadIdx.y; // row

	if (x >= BCOL || y >= AROW) {
		return;
	}

	int idx = (y * BCOL) + x;  // A[y][x] --> y * BCOL + x
	int tmp = 0;
	size_t k;
	for (k = 0; k < ACOL; k++) {
		tmp += A[y * ACOL + k] * B[k * BCOL + x]; //A[y][k] + B[k][x];
	}
	C[idx] = tmp; // C[y][x]   
}


int main() {
	int (*a2d)[ACOL], (*b2d)[BCOL], (*c2d)[CCOL];
	size_t Asize, Bsize, Csize;
	int *a2d_c, *b2d_c, *c2d_c;
	cudaError_t err;


	Asize = AROW * sizeof(*a2d);
	Bsize = BROW * sizeof(*b2d);
	Csize = CROW * sizeof(*c2d);

	a2d = (int(*)[ACOL])malloc(Asize);
	b2d = (int(*)[BCOL])malloc(Bsize);
	c2d = (int(*)[CCOL])malloc(Csize);

	initArray((int*)a2d, AROW * ACOL);
	initArray((int*)b2d, BROW * BCOL);

  cudaMalloc(&a2d_c, Asize);
  cudaMalloc(&b2d_c, Bsize);
  cudaMalloc(&c2d_c, Csize);

  cudaMemcpy(a2d_c, a2d, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(b2d_c, b2d, Bsize, cudaMemcpyHostToDevice);

	dim3 n_blocks2d;
	dim3 n_threads2d(16, 4);

	// c2d[y][x]; y -> row, x -> column

	n_blocks2d.x = CEIL_DIV(CCOL, 16); // column
	n_blocks2d.y = CEIL_DIV(CROW, 4); // row

	matmul2D<<<n_blocks2d, n_threads2d>>>(a2d_c, b2d_c, c2d_c);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 2D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}
	cudaDeviceSynchronize(); // wait for all blocks to execute

	cudaMemcpy(c2d, c2d_c, Csize, cudaMemcpyDeviceToHost);

#ifdef VERIFY
	size_t i, j, k;
	for (i = 0; i < CROW; i++) {
		for (j = 0; j < CCOL; j++) {
			int tmp = 0;
			for (k = 0; k < ACOL; k++) {
				tmp += a2d[i][k] * b2d[k][j];
			}
			assert(c2d[i][j] == tmp);
		}
	}
#endif
	
	cudaFree(a2d_c);
	cudaFree(b2d_c);
	cudaFree(c2d_c);
	free(a2d);
	free(b2d);
	free(c2d);
	
	return 0;
}
