// Philip Carr
// CS 171 Assignment 0 Part 1
// October 10, 2018
// object.h

#include <vector>
using namespace std;

/**
 * vertex struct to store object vertices. Vertices are (x, y, z) points.
 */
struct vertex {
    double x;
    double y;
    double z;
};

/**
 * face struct to store object faces. Faces are triangular regions determined by
 * the vertices v1, v2, and v3.
 */
struct face {
    int v1;
    int v2;
    int v3;
};

/**
 * object struct to store vector of vertices and vector of faces that define the
 * object.
 */
struct object {
    vector<vertex> vertex_vector;
    vector<face> face_vector;
};
