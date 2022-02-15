// Philip Carr
// CS 171 Assignment 0 Part 3
// October 10, 2018
// object.h

#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/**
 *  vertex struct to store object vertices. Vertices are (x, y, z) points.
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
 * object struct to store an object's vector of vertices, vector of faces and
 * the objects' corresponding transformations
 */
struct object {
    vector<vertex> vertex_vector;
    vector<face> face_vector;
    vector<Matrix4d> transform_vector;
};
