
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

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
	int (*A)[W][H], (*B)[W][H], (*C)[W][H];
	size_t i, j, k, size3D;

	int *A_c, *B_c, *C_c;
	cudaError_t err;

	size3D = L * sizeof(*A);

	A = (int(*)[W][H])malloc(size3D);
	B = (int(*)[W][H])malloc(size3D);
	C = (int(*)[W][H])malloc(size3D);

	initArray((int*)A, L * W * H);
	initArray((int*)B, L * W * H);
  cudaMalloc(&A_c, size3D);
  cudaMalloc(&B_c, size3D);
  cudaMalloc(&C_c, size3D);

  cudaMemcpy(A_c, A, size3D, cudaMemcpyHostToDevice);
  cudaMemcpy(B_c, B, size3D, cudaMemcpyHostToDevice);

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

	cudaMemcpy(C, C_c, size3D, cudaMemcpyDeviceToHost);
	
	for (i = 0; i < L; i++) {
		for (j = 0; j < W; j++) {
			for (k = 0; k < H; k++) {
				assert(C[i][j][k] == A[i][j][k] + B[i][j][k]);
			}
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
