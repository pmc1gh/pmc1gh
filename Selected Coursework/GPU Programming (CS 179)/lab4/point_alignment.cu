/**
 * CUDA Point Alignment
 * George Stathopoulos, Jenny Lee, Mary Giambrone, 2019*/

#include <cstdio>
#include <stdio.h>
#include <fstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "helper_cuda.h"
#include <string>
#include <fstream>

#include "obj_structures.h"

// helper_cuda.h contains the error checking macros. note that they're called
// CUDA_CALL, CUBLAS_CALL, and CUSOLVER_CALL instead of the previous names

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main(int argc, char *argv[]) {

    if (argc != 4)
    {
        printf("Usage: ./point_alignment [file1.obj] [file2.obj] [output.obj]\n");
        return 1;
    }

    std::string filename, filename2, output_filename;
    filename = argv[1];
    filename2 = argv[2];
    output_filename = argv[3];

    std::cout << "Aligning " << filename << " with " << filename2 <<  std::endl;
    Object obj1 = read_obj_file(filename);
    std::cout << "Reading " << filename << ", which has " << obj1.vertices.size() << " vertices" << std::endl;
    Object obj2 = read_obj_file(filename2);

    std::cout << "Reading " << filename2 << ", which has " << obj2.vertices.size() << " vertices" << std::endl;
    if (obj1.vertices.size() != obj2.vertices.size())
    {
        printf("Error: number of vertices in the obj files do not match.\n");
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Loading in obj into vertex Array
    ///////////////////////////////////////////////////////////////////////////

    int point_dim = 4; // 3 spatial + 1 homogeneous
    int num_points = obj1.vertices.size();

    // in col-major
    float * x1mat = vertex_array_from_obj(obj1);
    float * x2mat = vertex_array_from_obj(obj2);

    ///////////////////////////////////////////////////////////////////////////
    // Point Alignment
    ///////////////////////////////////////////////////////////////////////////

    // TODO: Initialize cublas handle
    cublasHandle_t handle;

    cublasCreate(&handle);

    float * dev_x1mat;
    float * dev_x2mat;
    float * dev_xx4x4;
    float * dev_x1Tx2;

    // TODO: Allocate device memory and copy over the data onto the device
    // Hint: Use cublasSetMatrix() for copying

    cudaMalloc((void **) &dev_x1mat, num_points * point_dim * sizeof(float));
    cudaMalloc((void **) &dev_x2mat, num_points * point_dim * sizeof(float));
    cudaMalloc((void **) &dev_xx4x4, point_dim * point_dim * sizeof(float));
    cudaMalloc((void **) &dev_x1Tx2, point_dim * point_dim * sizeof(float));

    cublasSetMatrix(num_points, point_dim, sizeof(float), x1mat, num_points,
                    dev_x1mat, num_points);
    cublasSetMatrix(num_points, point_dim, sizeof(float), x2mat, num_points,
                    dev_x2mat, num_points);

    // Now, proceed with the computations necessary to solve for the linear
    // transformation.

    float one = 1;
    float zero = 0;

    // TODO: First calculate xx4x4 and x1Tx2
    // Following two calls should correspond to:
    //   xx4x4 = Transpose[x1mat] . x1mat
    //   x1Tx2 = Transpose[x1mat] . x2mat

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, point_dim, point_dim,
                num_points, &one, dev_x1mat, num_points, dev_x1mat, num_points,
                &zero, dev_xx4x4, point_dim);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, point_dim, point_dim,
                num_points, &one, dev_x1mat, num_points, dev_x2mat, num_points,
                &zero, dev_x1Tx2, point_dim);

    // TODO: Finally, solve the system using LU-factorization! We're solving
    //         xx4x4 . m4x4mat.T = x1Tx2   i.e.   m4x4mat.T = Inverse[xx4x4] . x1Tx2
    //
    //       Factorize xx4x4 into an L and U matrix, ie.  xx4x4 = LU
    //
    //       Then, solve the following two systems at once using cusolver's getrs
    //           L . temp  =  P . x1Tx2
    //       And then then,
    //           U . m4x4mat = temp
    //
    //       Generally, pre-factoring a matrix is a very good strategy when
    //       it is needed for repeated solves.

    // TODO: Make handle for cuSolver
    cusolverDnHandle_t solver_handle;

    cusolverDnCreate(&solver_handle);

    // TODO: Initialize work buffer using cusolverDnSgetrf_bufferSize
    float * work;
    int Lwork;

    cusolverDnSgetrf_bufferSize(solver_handle, point_dim, point_dim, dev_xx4x4,
                                point_dim, &Lwork);

    // TODO: compute buffer size and prepare memory

    cudaMalloc((void **) &work, Lwork * sizeof(float));

    // TODO: Initialize memory for pivot array, with a size of point_dim
    int * pivots;

    cudaMalloc((void **) &pivots, point_dim * sizeof(int));


    int *info;

    cudaMalloc((void **) &info, sizeof(int));


    // TODO: Now, call the factorizer cusolverDnSgetrf, using the above initialized data

    cusolverDnSgetrf(solver_handle, point_dim, point_dim, dev_xx4x4, point_dim,
                     work, pivots, info);

    // TODO: Finally, solve the factorized version using a direct call to cusolverDnSgetrs

    cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, point_dim, point_dim,
                     dev_xx4x4, point_dim, pivots, dev_x1Tx2, point_dim, info);


    // TODO: Destroy the cuSolver handle

    cusolverDnDestroy(solver_handle);


    // TODO: Copy final transformation back to host. Note that at this point
    // the transformation matrix is transposed
    float * out_transformation;

    out_transformation =
        (float *) malloc(point_dim * point_dim * sizeof(float));

    cudaMemcpy(out_transformation,
               dev_x1Tx2, point_dim * point_dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    // TODO: Don't forget to set the bottom row of the final transformation
    //       to [0,0,0,1] (right-most columns of the transposed matrix)

    out_transformation[3*point_dim] = 0;
    out_transformation[3*point_dim+1] = 0;
    out_transformation[3*point_dim+2] = 0;
    out_transformation[3*point_dim+3] = 1;

    // Print transformation in row order.
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << out_transformation[i * point_dim + j] << " ";
        }
        std::cout << "\n";
    }

    ///////////////////////////////////////////////////////////////////////////
    // Transform point and print output object file
    ///////////////////////////////////////////////////////////////////////////

    // TODO Allocate and Initialize data matrix
    float * dev_pt;

    cudaMalloc((void **) &dev_pt, num_points * point_dim * sizeof(float));
    cublasSetMatrix(num_points, point_dim, sizeof(float), x1mat, num_points,
                    dev_pt, num_points);

    // TODO Allocate and Initialize transformation matrix
    float * dev_trans_mat;

    cudaMalloc((void **) &dev_trans_mat, point_dim * point_dim * sizeof(float));
    cublasSetMatrix(point_dim, point_dim, sizeof(float), out_transformation,
                    point_dim, dev_trans_mat, point_dim);

    // TODO Allocate and Initialize transformed points
    float * dev_trans_pt;

    cudaMalloc((void **) &dev_trans_pt, point_dim * num_points * sizeof(float));

    float one_d = 1;
    float zero_d = 0;

    // TODO Transform point matrix
    //          (4x4 trans_mat) . (nx4 pointzx matrix)^T = (4xn transformed points)

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, point_dim, num_points,
                point_dim, &one_d, dev_trans_mat, point_dim, dev_pt, num_points,
                &zero_d, dev_trans_pt, point_dim);

    // So now dev_trans_pt has shape (4 x n)
    float * trans_pt;

    trans_pt = (float *) malloc(point_dim * num_points * sizeof(float));
    cublasGetMatrix(point_dim, num_points, sizeof(float), dev_trans_pt,
                    point_dim, trans_pt, point_dim);

    // get Object from transformed vertex matrix
    Object trans_obj = obj_from_vertex_array(trans_pt, num_points, point_dim,
                                             obj1);

    // print Object to output file
    std::ofstream obj_file (output_filename);
    print_obj_data(trans_obj, obj_file);

    // free CPU memory
    free(trans_pt);

    ///////////////////////////////////////////////////////////////////////////
    // Free Memory
    ///////////////////////////////////////////////////////////////////////////

    // TODO: Free GPU memory

    cudaFree(dev_x1mat);
    cudaFree(dev_x2mat);
    cudaFree(dev_xx4x4);
    cudaFree(dev_x1Tx2);

    cudaFree(work);
    cudaFree(pivots);
    cudaFree(info);

    cudaFree(dev_pt);
    cudaFree(dev_trans_mat);
    cudaFree(dev_trans_pt);

    cublasDestroy(handle);

    // TODO: Free CPU memory
    free(out_transformation);
    free(x1mat);
    free(x2mat);

}
