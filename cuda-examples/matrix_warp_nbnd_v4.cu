
// not handling unaligned boundaries
// tensor cores comp + loads on GPU cores
// 16 warps per block
// 16 x 16 tensor computation

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define THREADS_PER_BLOCK ((size_t)64)

//#define VERIFY 1
#define FACTOR 300

#define AROW ((THREADS_PER_BLOCK * FACTOR))
#define ACOL ((THREADS_PER_BLOCK * FACTOR))
#define BROW ACOL
#define BCOL ((THREADS_PER_BLOCK * FACTOR)) 
#define CROW AROW
#define CCOL BCOL

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

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

	__shared__ half tmpA[2][4][16][16];
	__shared__ half tmpB[2][4][16][16];
	int warp_id = threadIdx.x / 32;
	int wy = warp_id / 4;
	int wx = warp_id % 4;
	int tid = threadIdx.x % 32;
	int row, col;
	
	int _x = (blockIdx.x * 64 + wx * 16); // column
	int _y = (blockIdx.y * 64 + wy * 16); // row

	int k;

	int flat_idx_A = (32 * wx) + tid;
	int i_A = flat_idx_A / 8;
	int j_A = (flat_idx_A % 8) * 2;

	int flat_idx_B = (32 * wy) + tid;
	int i_B = flat_idx_B / 8;
	int j_B = (flat_idx_B % 8) * 2;
	
	row = (_y + i_A);
	col = j_A;
	*((float*)&tmpA[0][wy][i_A][j_A]) = *((float*)&A[row * ACOL + col]);

	row = i_B;
	col = _x + j_B;
	*((float*)&tmpB[0][wx][i_B][j_B]) = *((float*)&B[row * BCOL + col]);

	__syncthreads();

	

	wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> tmpAf;
	wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> tmpBf;
	wmma::fragment<wmma::accumulator, 16,16,16, float> tmpCf;

	wmma::fill_fragment(tmpCf, 0.0f);

	int read_idx = 0;
	int write_idx = 1;
	int t;

	for (k = 0; k < ACOL; k += 16) {
		wmma::load_matrix_sync(tmpAf, (half*)&tmpA[read_idx][wy], 16);
		wmma::load_matrix_sync(tmpBf, (half*)&tmpB[read_idx][wx], 16);

		if (k + 16 < ACOL) {
			row = (_y + i_A);
			col = k + 16 + j_A;
			*((float*)&tmpA[write_idx][wy][i_A][j_A]) = *((float*)&A[row * ACOL + col]);

			row = k + 16 + i_B;
			col = _x + j_B;
			*((float*)&tmpB[write_idx][wx][i_B][j_B]) = *((float*)&B[row * BCOL + col]);
		}

		wmma::mma_sync(tmpCf, tmpAf, tmpBf, tmpCf);
		__syncthreads();
		t = read_idx;
		read_idx = write_idx;
		write_idx = t;
	}
	wmma::store_matrix_sync((C + (_y * CCOL) + _x), tmpCf,  CCOL, wmma::mem_row_major);
}


int main() {
	half (*a2d)[ACOL], (*b2d)[BCOL];
	float (*c2d)[CCOL];
	size_t Asize, Bsize, Csize;
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
	dim3 n_threads2d(32*16, 1);

	// c2d[y][x]; y -> row, x -> column

	n_blocks2d.x = CEIL_DIV(CCOL, 16*4); // column
	n_blocks2d.y = CEIL_DIV(CROW, 16*4); // row

	matmul2D<<<n_blocks2d, n_threads2d>>>(a2d_c, b2d_c, c2d_c);
	cudaDeviceSynchronize(); // wait for all blocks to execute
	cudaMemcpy(c2d, c2d_c, Csize, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA error in 2D kernel: %s\n", cudaGetErrorString(err));
		exit(1);
	}

	//print_matrix((int*)c2d, "C");

#ifdef VERIFY
	size_t i, j, k;
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
#endif
	
	cudaFree(a2d_c);
	cudaFree(b2d_c);
	cudaFree(c2d_c);
	free(a2d);
	free(b2d);
	free(c2d);
	
	return 0;
}
