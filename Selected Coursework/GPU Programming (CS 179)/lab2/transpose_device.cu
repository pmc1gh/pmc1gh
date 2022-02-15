#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        // Writing to output is not coalesced. Each successive thread in a warp
        // writes to a successive row in the output matrix. Therefore, the total
        // number of bytes separating the first element written to by the first
        // thread and the last byte written to by the last thread is
        // ((n * i_{32} + j_{32} + 4) - (n * i_0 + j_0)) * 4. Per warp, since the
        // block size is (64, 16), each warp fits within one row of the block,
        // so for each warp threadIdx.y is constant. Since blockIdx.x and
        // blockIdx.y are also constant in a warp, j_{32} = j_0, and
        // i_{32} - i_0 = threadIdx.x_{32} - threadIdx.x_0 = 32 - 0 = 32.
        // Therefore, ((n * i_{32} + j_{32} + 4) - (n * i_0 + j_0)) * 4
        // = (n * (i_{32} - i_0) + 4) * 4 = 128n + 16 bytes. Therefore, per
        // warp, writing to the output matrix uses floor((128n + 16)/128) + 1
        // = (n + 1) 128-byte cache lines if the starting byte read by thread 0,
        // (n * i_0 + j_0) * 4, starts at the beginning of a new cache line or
        // is otherwise well-aligned with the cache line, or
        // (n + 2) 128-byte cache lines if the starting byte read by thread 0
        // is not well-aligned with the cache lines.
        // Number of 128-byte cache lines written to by the following write to
        // the output matrix: (n + 1) or (n + 2).

        // No bank conflicts are present here because input and output are in
        // global memory.
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // In this implementation, there are no bank conflicts because of the
    // padding of the shared memory, and all reads and writes are coalesced
    // (because each successive thread reads/writes to each successive element
    // in the input and output matrices along each column of the matrices.)

    // By padding the shared memory by +1 in the effective row size of the
    // data (matrix) array, no bank conflicts occur because having an effective
    // 65 elements per "row" in data offsets the alignment of banks in the
    // shared memory by 1, which allows threads to read and write with separate
    // banks.
    __shared__ float data[64*65];

    // i and j are the (start) column and row, respectively, in the input
    // matrix for the current thread.
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_k = 4;

    // Transpose the matrix block while writing to data.
    for (int k = 0; k < end_k; k++)
        data[4 * threadIdx.y + k + 64 * threadIdx.x] = input[i + n * (j + k)];

    // Make sure that all the threads are done writing to data before Writing
    // to output.
    __syncthreads();

    // it and jt are the transposed (start) column and row, respectively, in the output (transposed)
    // matrix.
    const int it = threadIdx.x + 64 * blockIdx.y;
    int jt = 4 * threadIdx.y + 64 * blockIdx.x;

    // Write the transposed block matrix in data to output in the transposed
    // row and column of the output (transposed) matrix.
    for (int k = 0; k < end_k; k++)
        output[it + n * (jt + k)] =
            data[threadIdx.x + 64 * (4 * threadIdx.y + k)];
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // Optimization technique used: loop unrolling.
    
    // In this implementation, there are no bank conflicts because of the
    // padding of the shared memory, and all reads and writes are coalesced
    // (because each successive thread reads/writes to each successive element
    // in the input and output matrices along each column of the matrices.)

    // By padding the shared memory by +1 in the effective row size of the
    // data (matrix) array, no bank conflicts occur because having an effective
    // 65 elements per "row" in data offsets the alignment of banks in the
    // shared memory by 1, which allows threads to read and write with separate
    // banks.
    __shared__ float data[64*65];

    // i and j are the (start) column and row, respectively, in the input
    // matrix for the current thread.
    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;

    // Transpose the matrix block while writing to data.
    data[4 * threadIdx.y + 0 + 64 * threadIdx.x] = input[i + n * (j + 0)];
    data[4 * threadIdx.y + 1 + 64 * threadIdx.x] = input[i + n * (j + 1)];
    data[4 * threadIdx.y + 2 + 64 * threadIdx.x] = input[i + n * (j + 2)];
    data[4 * threadIdx.y + 3 + 64 * threadIdx.x] = input[i + n * (j + 3)];

    // Make sure that all the threads are done writing to data before Writing
    // to output.
    __syncthreads();

    // it and jt are the transposed (start) column and row, respectively, in the output (transposed)
    // matrix.
    const int it = threadIdx.x + 64 * blockIdx.y;
    int jt = 4 * threadIdx.y + 64 * blockIdx.x;

    // Write the transposed block matrix in data to output in the transposed
    // row and column of the output (transposed) matrix.
    output[it + n * (jt + 0)] = data[threadIdx.x + 64 * (4 * threadIdx.y + 0)];
    output[it + n * (jt + 1)] = data[threadIdx.x + 64 * (4 * threadIdx.y + 1)];
    output[it + n * (jt + 2)] = data[threadIdx.x + 64 * (4 * threadIdx.y + 2)];
    output[it + n * (jt + 3)] = data[threadIdx.x + 64 * (4 * threadIdx.y + 3)];

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
