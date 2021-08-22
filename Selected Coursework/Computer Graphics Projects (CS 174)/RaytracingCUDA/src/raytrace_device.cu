// CS 179 Project: GPU-accelerated ray tracing
// Team members: Philip Carr and Thomas Leing
// June 10, 2019
// raytrace_device.cu

#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_header.cuh"
#include "structs.h"

// Check if the ray hit this axis-aligned bounding box.
CUDA_CALLABLE
bool doesIntersect(KDTreeCUDA* box, Ray ray) {
    double tmin = (box->xmin - ray.origin_x) / ray.direction_x;
    double tmax = (box->xmax - ray.origin_x) / ray.direction_x;
    if (tmin > tmax) {
        double temp = tmin;
        tmin = tmax;
        tmax = temp;
    }

    double tymin = (box->ymin - ray.origin_y) / ray.direction_y;
    double tymax = (box->ymax - ray.origin_y) / ray.direction_y;
    if (tymin > tymax) {
        double temp = tymin;
        tymin = tymax;
        tymax = temp;
    }

    if (tmin > tymax || tmax < tymin) return false;

    tmin = max(tmin, tymin);
    tmax = min(tmax, tymax);

    double tzmin = (box->zmin - ray.origin_z) / ray.direction_z;
    double tzmax = (box->zmax - ray.origin_z) / ray.direction_z;
    if (tzmin > tzmax) {
        double temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }

    if (tmin > tzmax || tmax < tzmin) return false;
    return true;
}

// Stack-based traversal of KDTree to determine the closest intersection
// point between the triangles of the tree and of the ray.
// Second half commented out -- might work on a version of CUDA which
// implements recursion properly.
__device__
Intersection cudaFindClosestIntersection(KDTreeCUDA* totalTree, Ray ray) {

#if 1
    Intersection noHit;
    noHit.intersection_ray.is_intersect = false;
    noHit.obj_num = -1;
    Intersection closestIntersection = noHit;
    double closestIntersectionDist = 999999999;

    // Stack to simulate recursion.
    KDTreeCUDA** stack = (KDTreeCUDA**) malloc(30 * sizeof(KDTreeCUDA*));
    stack[0] = totalTree;

    int depth = 0;
    while(depth >= 0) {
        KDTreeCUDA* tree = stack[depth--];
        if (tree == NULL) {continue;}
        if (!doesIntersect(tree, ray)) {continue;}

        if (tree->numTris == 0){
            // Interior tree node: keep traversing.
            if (tree->left != NULL) {
                stack[++depth] = tree->left;
            }
            if (tree->right != NULL) {
                stack[++depth] = tree->right;
            }
        }
        else {
            // We've hit a leaf!  Check all the triangles.
            for (int j = 0; j < (int) tree->numTris; ++j) {
                //cout << j << endl;
                Intersection intersection;

                float xp = ray.origin_x;
                float yp = ray.origin_y;
                float zp = ray.origin_z;

                float xd = ray.direction_x;
                float yd = ray.direction_y;
                float zd = ray.direction_z;

                float xa = tree->tris[j].v1.x;
                float ya = tree->tris[j].v1.y;
                float za = tree->tris[j].v1.z;

                float xb = tree->tris[j].v2.x;
                float yb = tree->tris[j].v2.y;
                float zb = tree->tris[j].v2.z;

                float xc = tree->tris[j].v3.x;
                float yc = tree->tris[j].v3.y;
                float zc = tree->tris[j].v3.z;


                // 3x3 matrix inversion by hand, since no Eigen allowed on CUDA
                double detA =
                    -(xd * yb * za) + (xd * yc * za) + (xb * yd * za)
                    -(xc * yd * za) + (xd * ya * zb) - (xd * yc * zb)
                    -(xa * yd * zb) + (xc * yd * zb) - (xd * ya * zc)
                    +(xd * yb * zc) + (xa * yd * zc) - (xb * yd * zc)
                    -(xb * ya * zd) + (xc * ya * zd) + (xa * yb * zd)
                    -(xc * yb * zd) - (xa * yc * zd) + (xb * yc * zd);

                double A_inv11_d = (-yd * za + yd * zc + ya * zd - yc * zd) / detA;
                double A_inv12_d = (xd * za - xd * zc - xa * zd + xc * zd) / detA;
                double A_inv13_d = (-xd * ya + xd * yc + xa * yd - xc * yd) / detA;

                double A_inv21_d = (yd * za - yd * zb - ya * zd + yb * zd) / detA;
                double A_inv22_d = (-xd * za + xd * zb + xa * zd - xb * zd) / detA;
                double A_inv23_d = (xd * ya - xd * yb - xa * yd + xb * yd) / detA;

                double A_inv31_d = (-yb * za + yc * za + ya * zb - yc * zb - ya * zc + yb * zc) / detA;
                double A_inv32_d = (xb * za - xc * za - xa * zb + xc * zb + xa * zc - xb * zc) / detA;
                double A_inv33_d = (-xb * ya + xc * ya + xa * yb - xc * yb - xa * yc + xb * yc) / detA;

                /*
                Matrix3d A_inv_d;
                A_inv_d << A_inv11_d, A_inv12_d, A_inv13_d,
                         A_inv21_d, A_inv22_d, A_inv23_d,
                         A_inv31_d, A_inv32_d, A_inv33_d;


                Matrix3f A;
                A << xa - xb, xa - xc, xd,
                     ya - yb, ya - yc, yd,
                     za - zb, za - zc, zd;

                Matrix3f A_inv = A.inverse();

                Vector3f b;
                b << xa - xp, ya - yp, za - zp;

                Vector3f soln;
                soln = A_inv * b;

                float beta = soln(0);
                float gamma = soln(1);
                float t = soln(2);
                */

                // [beta          [[A11 A12 A13]       [xa - xp
                //  gamma   =      [A21 A22 A23]   *    ya - yp
                //    t  ]         [A31 A32 A33]]       za - zp]

                float beta  = A_inv11_d * (xa - xp) + A_inv12_d * (ya - yp) + A_inv13_d * (za - zp);
                float gamma = A_inv21_d * (xa - xp) + A_inv22_d * (ya - yp) + A_inv23_d * (za - zp);
                float t     = A_inv31_d * (xa - xp) + A_inv32_d * (ya - yp) + A_inv33_d * (za - zp);

                // Normal interpolation across triangle.
                if (beta > 0 && gamma > 0 && beta + gamma < 1) {
                    //cout << " " << xa << " " << ya << " " << za << endl;
                    //cout << " " << xb << " " << yb << " " << zb << endl;
                    //cout << " " << xc << " " << yc << " " << zc << endl;
                    float alpha = 1 - beta - gamma;
                    Ray intersection_ray;
                    intersection_ray.origin_x =
                        ray.origin_x + t * ray.direction_x;
                    intersection_ray.origin_y =
                        ray.origin_y + t * ray.direction_y;
                    intersection_ray.origin_z =
                        ray.origin_z + t * ray.direction_z;

                    intersection_ray.direction_x =
                        alpha * tree->normals[j].v1.x
                        + beta * tree->normals[j].v2.x
                        + gamma * tree->normals[j].v3.x;
                    intersection_ray.direction_y =
                        alpha * tree->normals[j].v1.y
                        + beta * tree->normals[j].v2.y
                        + gamma * tree->normals[j].v3.y;
                    intersection_ray.direction_z =
                        alpha * tree->normals[j].v1.z
                        + beta * tree->normals[j].v2.z
                        + gamma * tree->normals[j].v3.z;

                    /*
                    double normalization = intersection_ray.direction_x * intersection_ray.direction_x + intersection_ray.direction_y * intersection_ray.direction_y + intersection_ray.direction_z * intersection_ray.direction_z;
                    intersection_ray.direction_x /= normalization;
                    intersection_ray.direction_y /= normalization;
                    intersection_ray.direction_z /= normalization;
                    */

                    intersection_ray.is_intersect = true;

                    intersection.intersection_ray = intersection_ray;
                    intersection.obj_num = 0;

                    // Is this intersection the closest to the ray origin so far?
                    double dist = sqrt(pow(intersection_ray.origin_x - ray.origin_x, 2)
                                     + pow(intersection_ray.origin_y - ray.origin_y, 2)
                                     + pow(intersection_ray.origin_z - ray.origin_z, 2));

                    //cout << dist << endl;
                    //cout << intersection_ray.origin_x << " " << intersection_ray.origin_y << " " << intersection_ray.origin_z << endl;
                    if (dist < closestIntersectionDist) {
                        closestIntersection = intersection;
                        closestIntersectionDist = dist;
                    }
                }
            }
        }
    }
    return closestIntersection;




#else

    // Sad recursive code :(
    // Many exciting and mysterious bugs await if you dare to investigate


    //printf("cudaCall: %d %d\n", threadIdx.x, blockIdx.x);
    printf("nil?: %p, %d\n", tree, tree->numTris);
    printf("%d\n", depth);
    // Recurse through the KD tree to find the closest intersection of the camera ray with all of the triangles in the KD tree.
    Intersection noHit;
    noHit.intersection_ray.is_intersect = false;
    noHit.obj_num = -1;
    if(tree == NULL) {printf("tree == NULL\n"); return noHit;}
    if (!doesIntersect(tree, ray)) {printf("no aabb intersect\n");return noHit;}

    if (depth < 4 && tree->numTris == 0) {
        printf("nil part 2?: %d\n", tree->numTris);
        //return noHit;
        if (tree->left == NULL) {
            return cudaFindClosestIntersection(tree->right, ray, depth+1);
        }
        if (tree->right == NULL) {
            return cudaFindClosestIntersection(tree->left, ray, depth+1);
        }
        Intersection leftIntersection = cudaFindClosestIntersection(tree->left, ray, depth+1);
        Intersection rightIntersection = cudaFindClosestIntersection(tree->right, ray, depth+1);

        // If either is a noHit, return the other.
        if (leftIntersection.obj_num == -1) return rightIntersection;
        if (rightIntersection.obj_num == -1) return leftIntersection;

        // Return the closest of leftIntersection and rightIntersection, since both are valid.
        float iray_x = leftIntersection.intersection_ray.origin_x;
        float iray_y = leftIntersection.intersection_ray.origin_y;
        float iray_z = leftIntersection.intersection_ray.origin_z;
        double distanceLeft = sqrt(pow(iray_x - ray.origin_x, 2)
                              + pow(iray_y - ray.origin_y, 2)
                              + pow(iray_z - ray.origin_z, 2));
        iray_x = rightIntersection.intersection_ray.origin_x;
        iray_y = rightIntersection.intersection_ray.origin_y;
        iray_z = rightIntersection.intersection_ray.origin_z;
        double distanceRight = sqrt(pow(iray_x - ray.origin_x, 2)
                              + pow(iray_y - ray.origin_y, 2)
                              + pow(iray_z - ray.origin_z, 2));
        if (distanceLeft < distanceRight) return leftIntersection;
        return rightIntersection;
    }

    // Otherwise, we've hit a leaf node.  Look through all of
    // the contents of the tree and see which the closest is,
    // if any.
    //printf("tris or normals?: %p, %p\n", tree->tris, tree->normals);
    Intersection closestIntersection = noHit;
    double closestIntersectionDist = 999999999;
    for (int j = 0; j < (int) tree->numTris; ++j) {
        //cout << j << endl;
        Intersection intersection;

        float xp = ray.origin_x;
        float yp = ray.origin_y;
        float zp = ray.origin_z;

        float xd = ray.direction_x;
        float yd = ray.direction_y;
        float zd = ray.direction_z;

        float xa = tree->tris[j].v1.x;
        float ya = tree->tris[j].v1.y;
        float za = tree->tris[j].v1.z;

        float xb = tree->tris[j].v2.x;
        float yb = tree->tris[j].v2.y;
        float zb = tree->tris[j].v2.z;

        float xc = tree->tris[j].v3.x;
        float yc = tree->tris[j].v3.y;
        float zc = tree->tris[j].v3.z;


        // 3x3 matrix inversion by hand, since no Eigen allowed on CUDA
        double detA =
            -(xd * yb * za) + (xd * yc * za) + (xb * yd * za)
            -(xc * yd * za) + (xd * ya * zb) - (xd * yc * zb)
            -(xa * yd * zb) + (xc * yd * zb) - (xd * ya * zc)
            +(xd * yb * zc) + (xa * yd * zc) - (xb * yd * zc)
            -(xb * ya * zd) + (xc * ya * zd) + (xa * yb * zd)
            -(xc * yb * zd) - (xa * yc * zd) + (xb * yc * zd);

        double A_inv11_d = (-yd * za + yd * zc + ya * zd - yc * zd) / detA;
        double A_inv12_d = (xd * za - xd * zc - xa * zd + xc * zd) / detA;
        double A_inv13_d = (-xd * ya + xd * yc + xa * yd - xc * yd) / detA;

        double A_inv21_d = (yd * za - yd * zb - ya * zd + yb * zd) / detA;
        double A_inv22_d = (-xd * za + xd * zb + xa * zd - xb * zd) / detA;
        double A_inv23_d = (xd * ya - xd * yb - xa * yd + xb * yd) / detA;

        double A_inv31_d = (-yb * za + yc * za + ya * zb - yc * zb - ya * zc + yb * zc) / detA;
        double A_inv32_d = (xb * za - xc * za - xa * zb + xc * zb + xa * zc - xb * zc) / detA;
        double A_inv33_d = (-xb * ya + xc * ya + xa * yb - xc * yb - xa * yc + xb * yc) / detA;

        /*
        Matrix3d A_inv_d;
        A_inv_d << A_inv11_d, A_inv12_d, A_inv13_d,
                 A_inv21_d, A_inv22_d, A_inv23_d,
                 A_inv31_d, A_inv32_d, A_inv33_d;


        Matrix3f A;
        A << xa - xb, xa - xc, xd,
             ya - yb, ya - yc, yd,
             za - zb, za - zc, zd;

        Matrix3f A_inv = A.inverse();

        Vector3f b;
        b << xa - xp, ya - yp, za - zp;

        Vector3f soln;
        soln = A_inv * b;

        float beta = soln(0);
        float gamma = soln(1);
        float t = soln(2);
        */

        // [beta          [[A11 A12 A13]       [xa - xp
        //  gamma   =      [A21 A22 A23]   *    ya - yp
        //    t  ]         [A31 A32 A33]]       za - zp]

        float beta  = A_inv11_d * (xa - xp) + A_inv12_d * (ya - yp) + A_inv13_d * (za - zp);
        float gamma = A_inv21_d * (xa - xp) + A_inv22_d * (ya - yp) + A_inv23_d * (za - zp);
        float t     = A_inv31_d * (xa - xp) + A_inv32_d * (ya - yp) + A_inv33_d * (za - zp);

        if (beta > 0 && gamma > 0 && beta + gamma < 1) {
            //cout << " " << xa << " " << ya << " " << za << endl;
            //cout << " " << xb << " " << yb << " " << zb << endl;
            //cout << " " << xc << " " << yc << " " << zc << endl;
            float alpha = 1 - beta - gamma;
            Ray intersection_ray;
            intersection_ray.origin_x =
                ray.origin_x + t * ray.direction_x;
            intersection_ray.origin_y =
                ray.origin_y + t * ray.direction_y;
            intersection_ray.origin_z =
                ray.origin_z + t * ray.direction_z;

            intersection_ray.direction_x =
                alpha * tree->normals[j].v1.x
                + beta * tree->normals[j].v2.x
                + gamma * tree->normals[j].v3.x;
            intersection_ray.direction_y =
                alpha * tree->normals[j].v1.y
                + beta * tree->normals[j].v2.y
                + gamma * tree->normals[j].v3.y;
            intersection_ray.direction_z =
                alpha * tree->normals[j].v1.z
                + beta * tree->normals[j].v2.z
                + gamma * tree->normals[j].v3.z;

            /*
            double normalization = intersection_ray.direction_x * intersection_ray.direction_x + intersection_ray.direction_y * intersection_ray.direction_y + intersection_ray.direction_z * intersection_ray.direction_z;
            intersection_ray.direction_x /= normalization;
            intersection_ray.direction_y /= normalization;
            intersection_ray.direction_z /= normalization;
            */

            intersection_ray.is_intersect = true;

            intersection.intersection_ray = intersection_ray;
            intersection.obj_num = 0;

            // Is this intersection the closest to the ray origin so far?
            double dist = sqrt(pow(intersection_ray.origin_x - ray.origin_x, 2)
                             + pow(intersection_ray.origin_y - ray.origin_y, 2)
                             + pow(intersection_ray.origin_z - ray.origin_z, 2));

            //cout << dist << endl;
            //cout << intersection_ray.origin_x << " " << intersection_ray.origin_y << " " << intersection_ray.origin_z << endl;
            if (dist < closestIntersectionDist) {
                closestIntersection = intersection;
                closestIntersectionDist = dist;
            }
        }
    }

    /*
    if(!closestIntersection.intersection_ray.is_intersect) {
        printf("noHit strikes again\n");
    }
    */

    return closestIntersection;
#endif
}

// Kernel for finding closest ray-object intersections for all given deviceRays.
__global__
void cudaFindClosestIntersectionKernel(KDTreeCUDA* deviceTree, Ray* deviceRays, Intersection* deviceStorage, int rows, int cols) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    while(idx < rows * cols) {
        // Fetch current ray for the pixel and find the closest intersection on the tree.
        Ray ray = deviceRays[idx];
        deviceStorage[idx] = cudaFindClosestIntersection(deviceTree, ray);
        idx += blockDim.x * gridDim.x;
    }
}

// Call into the kernel -- this is the interface for the raytrace.cpp file to call for intersection testing.
void cudaCallFindClosestIntersectionKernel(KDTreeCUDA* deviceTree, Ray* deviceRays, Intersection* deviceIntersections, int rows, int cols) {
    // Break pixel grid into 64x64 thread grid.
    cudaFindClosestIntersectionKernel<<<64, 64>>>(deviceTree, deviceRays, deviceIntersections, rows, cols);
}

// Kernel for helping build a KDTree struct in CUDA.
__global__
void cudaKDTreeLink(KDTreeCUDA* deviceTree, Tri* tris, Tri* normals, KDTreeCUDA* deviceTreeLeft, KDTreeCUDA* deviceTreeRight) {
    // Connect the complex datatypes in a KDTree struct.
    deviceTree->tris = tris;
    deviceTree->normals = normals;
    deviceTree->left = deviceTreeLeft;
    deviceTree->right = deviceTreeRight;
}

// Call into the kernel -- this is the interface for the raytrace.cpp file to call for finishing copying a KDTree.
void cudaCallKDTreeLink(KDTreeCUDA* deviceTree, Tri* tris, Tri* normals, KDTreeCUDA* deviceTreeLeft, KDTreeCUDA* deviceTreeRight) {
    KDTreeCUDA* hostTreeCUDA = (KDTreeCUDA*) malloc(sizeof(KDTreeCUDA));
    CUDA_CALL( cudaMemcpy(hostTreeCUDA, deviceTree, sizeof(KDTreeCUDA), cudaMemcpyDeviceToHost) );
    cudaKDTreeLink<<<1, 1>>>(deviceTree, tris, normals, deviceTreeLeft, deviceTreeRight);
}
