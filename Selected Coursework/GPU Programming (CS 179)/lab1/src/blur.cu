/*
 * CUDA blur
 * Kevin Yuh, 2014
 * Revised by Nailen Matschke, 2016
 * Revised by Loko Kung, 2018
 */

#include "blur.cuh"

#include <cstdio>
#include <cuda_runtime.h>

#include "cuda_header.cuh"

CUDA_CALLABLE
void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {

    // Use n_frames for out of bounds check.
    if (thread_index >= n_frames) {
        return;
    }

    if (thread_index < blur_v_size) {
        for (int j = 0; j <= thread_index; j++)
            gpu_out_data[thread_index] +=
                gpu_raw_data[thread_index - j] * gpu_blur_v[j];
    }
    else {
        for (int j = 0; j < blur_v_size; j++)
            gpu_out_data[thread_index] +=
                gpu_raw_data[thread_index - j] * gpu_blur_v[j];
    }
}

__global__
void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < n_frames) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(thread_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);

        thread_index += blockDim.x * gridDim.x;
    }
}


float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    float* gpu_raw_data;
    cudaMalloc((void**) &gpu_raw_data, n_frames * sizeof(float));
    cudaMemcpy(gpu_raw_data, raw_data, n_frames * sizeof(float),
               cudaMemcpyHostToDevice);

    float* gpu_blur_v;
    cudaMalloc((void**) &gpu_blur_v, blur_v_size * sizeof(float));
    cudaMemcpy(gpu_blur_v, blur_v, blur_v_size * sizeof(float),
               cudaMemcpyHostToDevice);

    float* gpu_out_data;
    cudaMalloc((void**) &gpu_out_data, n_frames * sizeof(float));

    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v,
        gpu_out_data, n_frames, blur_v_size);

    // Check for errors on kernel call
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err)
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    else
        fprintf(stderr, "No kernel error detected\n");

    cudaMemcpy(out_data, gpu_out_data, n_frames * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(gpu_raw_data);
    cudaFree(gpu_blur_v);
    cudaFree(gpu_out_data);

    // Stop the recording timer and return the computation time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
    return time_milli;
}
