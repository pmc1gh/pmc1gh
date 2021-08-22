// CS 179 Project: GPU-accelerated ray tracing
// Team members: Philip Carr and Thomas Leing
// June 10, 2019
// raytrace_device.cuh

#ifndef __RAYTRACE_DEVICE_CUH__
#define __RAYTRACE_DEVICE_CUH__

#include "structs.h"

bool doesIntersect(KDTreeCUDA* box, Ray ray);
Intersection cudaFindClosestIntersection(KDTreeCUDA* tree, Ray ray);
void cudaFindClosestIntersectionKernel(KDTreeCUDA* deviceTree, Ray* deviceRays, Intersection* deviceStorage, int rows);
void cudaCallFindClosestIntersectionKernel(KDTreeCUDA* deviceTree, Ray* deviceRays, Intersection* deviceIntersections, int rows, int cols);

void cudaKDTreeLink(KDTreeCUDA* deviceTree, Tri* tris, Tri* normals, KDTreeCUDA* deviceTreeLeft, KDTreeCUDA* deviceTreeRight);
void cudaCallKDTreeLink(KDTreeCUDA* deviceTree, Tri* tris, Tri* normals, KDTreeCUDA* deviceTreeLeft, KDTreeCUDA* deviceTreeRight);

#endif
