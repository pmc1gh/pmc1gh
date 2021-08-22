// CS 179 Project: GPU-accelerated ray tracing
// Team members: Philip Carr and Thomas Leing
// June 10, 2019
// structs.h

#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>

struct Vec3f
{
	float x, y, z;
};

struct Vertex
{
    float x, y, z;
	// int index;
};

struct Face
{
    int idx1, idx2, idx3;
};

struct Mesh_Data
{
    std::vector<Vertex*> *vertices;
    std::vector<Face*> *faces;
};

struct Ray {
    // Boolean to tell if ray is actually represents an intersection.
    bool is_intersect;

    float origin_x;
    float origin_y;
    float origin_z;

    float direction_x;
    float direction_y;
    float direction_z;
};

struct Intersection {
    Ray intersection_ray;
    int obj_num;
};

enum SplitDir { x, y, z };

struct Tri {
    Vertex v1;
    Vertex v2;
    Vertex v3;
};

/*
typedef struct KDTree {
    // Tree structure
    struct KDTree* left;
    struct KDTree* right;

    // Each level of the tree has the same splitting direction, which alternates
    // x -> y -> z -> x -> ...
    // for the purpose of making the tree well-behaved I guess
    SplitDir dir;

    // Axis-aligned bounding box (AABB)
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;

    // Faces in AABB (only for leaf nodes)
    vector<Tri> tris;
    vector<Tri> normals;
} KDTree;
*/


typedef struct KDTreeCUDA {
    // Tree structure
    struct KDTreeCUDA* left;
    struct KDTreeCUDA* right;

    // Each level of the tree has the same splitting direction, which alternates
    // x -> y -> z -> x -> ...
    // for the purpose of making the tree well-behaved I guess
    SplitDir dir;

    // Axis-aligned bounding box (AABB)
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;

    // Faces in AABB (only for leaf nodes)
    // Need to be arrays and not vectors so that they can be used on GPU.
    Tri* tris;
    Tri* normals;
    int numTris;
} KDTreeCUDA;


typedef struct Stack {
    struct Stack* next;
    KDTreeCUDA* tree;
} Stack;


#endif
