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

    // printf("padded_length = %d\n", padded_length);
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

// Sequential addressing kernel.
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

    unsigned int thread_id = threadIdx.x;
    unsigned int out_data_i= blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int max_data_i = blockIdx.x;

    __syncthreads();

    while (out_data_i < padded_length) {
        // Reduction in shared memory rather than global memory.
        extern __shared__ float sdata[];
        sdata[thread_id] = abs(out_data[out_data_i].x);

        // This access pattern avoids bank conflicts because each thread
        // reads shared memory locations that are unique to the thread using the
        // appropriate stride (stride decreases by 2 and only threads that
        // have thread_id less than s continue with computation).

        // This reduction performs sequential addressing, which avoids bank
        // conflicts.
        for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
            if (thread_id < s) {
                atomicMax(&sdata[thread_id], sdata[thread_id + s]);
            }
            __syncthreads();
        }

        // Write the result for this block to global memory.
        if(thread_id == 0){
            max_data[max_data_i].x = sdata[0];
        }

        // Increment indices to process next set of blocks.
        out_data_i += blockDim.x * gridDim.x;
        max_data_i += gridDim.x;
        __syncthreads();
    }

    // Write the maximum value of max_data, which from the reduction should be
    // at index 0, to max_abs_val.
    atomicMax(max_abs_val, max_data[0].x);
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

    float norm = *max_abs_val;

    while (thread_index < padded_length) {
        // Do computation for this thread index
        out_data[thread_index].x /= norm;
        out_data[thread_index].y /= norm;

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
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v,
        out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        cufftComplex *max_data,
        float *max_abs_val,
        const unsigned int padded_length) {

    /* TODO 2: Call the max-finding kernel. */
    int current_padded_length = padded_length;
    cufftComplex *swap;

    // Perform reduction for as many times threadsPerBlock within value of
    // current_padded_length.
    while (current_padded_length > threadsPerBlock) {
        // Kernel writes max values to max_data.
        cudaMaximumKernel<<<blocks, threadsPerBlock,
                            threadsPerBlock * 2 * sizeof(float)>>>(out_data, max_data,
                          max_abs_val, current_padded_length);

        // Swap arrays for next iteration.
        swap = max_data;
        max_data = out_data;
        out_data = swap;

        // Divide current_padded_length by threadsPerBlock, then add 1 if there
        // is any remainder from the division.
        current_padded_length = current_padded_length / threadsPerBlock
            + (int) (current_padded_length % threadsPerBlock != 0);
    }

    // If the padded length is still positive then run the kenerl one more time.
    if (current_padded_length > 1) {
        cudaMaximumKernel<<<blocks, threadsPerBlock,
                            threadsPerBlock * 2 * sizeof(float)>>>(out_data, max_data,
                          max_abs_val, current_padded_length);
    }
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
