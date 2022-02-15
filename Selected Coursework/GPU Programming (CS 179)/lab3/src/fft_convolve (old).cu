/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/*
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source:
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v,
    cufftComplex *out_data,
    int padded_length) {


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response.

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them.

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
    // ATTEMPTED
    printf("padded_length = %d\n", padded_length);
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    float x_raw;
    float y_raw;

    float x_imp;
    float y_imp;

    while (thread_index < padded_length) {
        // Do computation for this thread index
        x_raw = raw_data[thread_index].x;
        y_raw = raw_data[thread_index].y;

        x_imp = impulse_v[thread_index].x;
        y_imp = impulse_v[thread_index].y;

        out_data[thread_index].x =
            (x_raw * x_imp - y_raw * y_imp) / padded_length;
        out_data[thread_index].y =
            (x_raw * y_imp + y_raw * x_imp) / padded_length;

        thread_index += blockDim.x * gridDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, cufftComplex *max_data,
                  float *max_abs_val, int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others.

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    // ATTEMPTED
    // __shared__ float data[2048];
    //
    // int max_pow_2 = 1;
    // while (max_pow_2 < padded_length) {
    //     max_pow_2 *= 2;
    // }
    //
    // uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    //
    // while (thread_index < max_pow_2) {
    //     // Do computation for this thread index
    //
    //     thread_index += blockDim.x * gridDim.x;
    //
    // }
    extern __shared__ cufftComplex sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = out_data[i];
    __syncthreads();

    float *maxPtr;
    malloc(&maxPtr, sizeof(float));

    // do reduction in shared mem
    for (unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            *maxPtr = sdata[tid].x;
            atomicMax(maxPtr, sdata[tid + s].x);
            sdata[tid].x = *maxPtr;
        }
        __syncthreads();
    }
    cudaFree(maxPtr);

    // write result for this block to global mem
    if (tid == 0) {
        max_data[blockIdx.x] = sdata[0];
    }
    return;
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val.

    This kernel should be quite short.
    */
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < padded_length) {
        // Do computation for this thread index
        out_data[thread_index].x /= *max_abs_val;
        out_data[thread_index].y /= *max_abs_val;

        thread_index += blockDim.x * gridDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {

    /* TODO: Call the element-wise product and scaling kernel. */
    // ATTEMPTED
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v,
        out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {


    /* TODO 2: Call the max-finding kernel. */
    int current_padded_length = padded_length;
    cufftComplex *max_data, *swap_data;
    cudaMalloc(&max_data,
               (padded_length / threadsPerBlock + threadsPerBlock)
               * sizeof(cufftComplex));
    cudaMemset(max_data, 0,
               (padded_length / threadsPerBlock + threadsPerBlock)
               * sizeof(cufftComplex));

    while (current_padded_length > threadsPerBlock) {
        cudaMaximumKernel<<<blocks, threadsPerBlock>>>(out_data, max_data,
            max_abs_val, current_padded_length);

        current_padded_length = current_padded_length / threadsPerBlock
            + (int) (current_padded_length % threadsPerBlock != 0);

        swap_data = max_data;
        max_data = out_data;
        out_data = swap_data;
    }
    if (current_padded_length > 1) {
        cudaMaximumKernel<<<1, current_padded_length>>>(out_data, max_data,
            max_abs_val, current_padded_length);
    }
    *max_abs_val = max_data[0].x;
    cudaFree(max_data);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val,
        padded_length);
}
