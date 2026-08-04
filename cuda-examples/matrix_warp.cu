
// handling unaligned boundaries
// only tensor cores computation
// one warp per block
// 16 x 16 tensor computation

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define THREADS_PER_BLOCK ((size_t)64)


#define AROW ((THREADS_PER_BLOCK * 31) + 2)
#define ACOL ((THREADS_PER_BLOCK * 15) + 4)
#define BROW ACOL
#define BCOL ((THREADS_PER_BLOCK * 2) + 6) 
#define CROW AROW
#define CCOL BCOL

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define CEIL_DIV1(x, y) (((x) + (1) - 1) / (y))

void
print_matrix(half *mat, const char *str) {
	int i, j;

	printf("Printing %s ...\n", str);
	for (i = 0; i < AROW; i++) {
		for (j = 0; j < ACOL; j++) {
			float f = __half2float(mat[i*ACOL+j]);
			printf("%f ", f);
		}
		printf("\n");
	}
}

static void initArray(half *arr, size_t x, size_t y) {
	size_t i, j;
	for (i = 0; i < x; i++) {
		for (j = 0; j < y; j++) {
			arr[(i *  y) + j] = __float2half((i + j + 13) % 4);
		}
	}
}

__global__ void matmul2D(half *A, half *B, float *C) {
	__shared__ half tmpA[16][16];
	__shared__ float tmp[16][16];

	int x = (blockIdx.x * 16); // column
	int y = (blockIdx.y * 16); // row
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int k, j;

	assert(ty == 0);
	wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> tmpAf;
	wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> tmpBf;
	wmma::fragment<wmma::accumulator, 16,16,16, float> tmpCf;

	wmma::fill_fragment(tmpCf, 0.0f);

	for (k = 0; k < ACOL; k += 16) {

		if (((k + 16) > ACOL || (y + 16) > AROW)) {
			int start_row = y;
			int start_col = k;
			int s = tx / 16;
			int i = tx % 16;
			int j_start = s * 8;
			int j_end = (s+1) * 8;

			// ON A100 cp.async may help here
			for (j = j_start; j < j_end; j++) {
				if ((start_row + i) < AROW && (start_col + j) < ACOL) {
					tmpA[i][j] = A[((start_row+i)*ACOL) + (start_col+j)];
				}
				else {
					tmpA[i][j] = 0;
				}
			}
			__syncthreads();
			wmma::load_matrix_sync(tmpAf, (half*)tmpA, 16);
		}
		else {
			// WARNING:: mptr must be 16-byte aligned
			wmma::load_matrix_sync(tmpAf, (A + (y * ACOL) + k), ACOL);
		}
		if (((k + 16) > BROW || (x + 16) > BCOL)) {
			int start_row = k;
			int start_col = x;
			int s = tx / 16;
			int i = tx % 16;
			int j_start = s * 8;
			int j_end = (s+1) * 8;

			// ON A100 cp.async may help here
			for (j = j_start; j < j_end; j++) {
				if ((start_row + i) < BROW && (start_col + j) < BCOL) {
					tmpA[i][j] = B[((start_row+i)*BCOL) + (start_col+j)];
				}
				else {
					tmpA[i][j] = 0;
				}
			}
			__syncthreads();
			wmma::load_matrix_sync(tmpBf, (half*)tmpA, 16);
		}
		else {
			// WARNING:: mptr must be 16-byte aligned
			wmma::load_matrix_sync(tmpBf, (B + (k * BCOL) + x), BCOL);
		}

		wmma::mma_sync(tmpCf, tmpAf, tmpBf, tmpCf);
	}
	
	if ((y + 16 > CROW || x + 16 > CCOL)) {
		wmma::store_matrix_sync((float*)tmp, tmpCf,  16, wmma::mem_row_major);
		int start_row = y;
		int start_col = x;
		int s = tx / 16;
		int i = tx % 16;
		int j_start = s * 8;
		int j_end = (s+1) * 8;

		for (j = j_start; j < j_end; j++) {
			if ((start_row + i) < CROW && (start_col + j) < CCOL) {
				C[((start_row + i) * CCOL) + (start_col + j)] = tmp[i][j];
			}
		}
	}
	else {
	  // WARNING:: mptr must be 16-byte aligned
		wmma::store_matrix_sync((C + (y * CCOL) + x), tmpCf,  CCOL, wmma::mem_row_major);
	}
}


int main() {
	half (*a2d)[ACOL], (*b2d)[BCOL];
	float (*c2d)[CCOL];
	size_t i, j, k, Asize, Bsize, Csize;
	half *a2d_c, *b2d_c;
	float *c2d_c;
	cudaError_t err;


	Asize = AROW * sizeof(*a2d);
	Bsize = BROW * sizeof(*b2d);
	Csize = CROW * sizeof(*c2d);

	a2d = (half(*)[ACOL])malloc(Asize);
	b2d = (half(*)[BCOL])malloc(Bsize);
	c2d = (float(*)[CCOL])malloc(Csize);

	initArray((half*)a2d, AROW, ACOL);
	initArray((half*)b2d, BROW, BCOL);

  cudaMalloc(&a2d_c, Asize);
  cudaMalloc(&b2d_c, Bsize);
  cudaMalloc(&c2d_c, Csize);

  cudaMemcpy(a2d_c, a2d, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(b2d_c, b2d, Bsize, cudaMemcpyHostToDevice);
	cudaMemset(c2d_c, 0, Csize);

	//print_matrix((int*)a2d, "A");
	//print_matrix((int*)b2d, "B");

	dim3 n_blocks2d;
	dim3 n_threads2d(32, 1);

	// c2d[y][x]; y -> row, x -> column

	n_blocks2d.x = CEIL_DIV(CCOL, 16); // column
	n_blocks2d.y = CEIL_DIV(CROW, 16); // row

	matmul2D<<<n_blocks2d, n_threads2d>>>(a2d_c, b2d_c, c2d_c);
	cudaDeviceSynchronize(); // wait for all blocks to execute
	cudaMemcpy(c2d, c2d_c, Csize, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 2D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	//print_matrix((int*)c2d, "C");

	for (i = 0; i < CROW; i++) {
		for (j = 0; j < CCOL; j++) {
			float tmp = 0;
			for (k = 0; k < ACOL; k++) {
				tmp += __half2float(a2d[i][k]) * __half2float(b2d[k][j]);
			}
			//assert(c2d[i][j] == tmp);
			if (c2d[i][j] != tmp) {
				printf("%zd %zd %f %f\n", i, j, tmp, c2d[i][j]);
			}
		}
	}
	
	cudaFree(a2d_c);
	cudaFree(b2d_c);
	cudaFree(c2d_c);
	free(a2d);
	free(b2d);
	free(c2d);
	
	return 0;
}
