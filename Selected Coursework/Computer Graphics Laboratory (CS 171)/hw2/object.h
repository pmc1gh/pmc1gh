// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// object.h

#ifndef object_h
#define object_h

#include <vector>
#include <Eigen/Dense>

#include "pixel.h"

using namespace std;
using namespace Eigen;

/**
 * vertex struct to store object vertices. Vertices are (x, y, z) points.
 */
struct vertex {
    float x;
    float y;
    float z;
};

/**
 * vertex normal vector struct used to store object vertex normal vectors.
 * Normal vectors (normals) are defined by their x, y, and z-components.
 */
struct vnorm {
    float x;
    float y;
    float z;
};

/**
 * ndc_vertex struct to store object's (Cartesian) NDC vertices.
 */
struct ndc_vertex {
    float x;
    float y;
    float z;
};

/**
 * s_coord struct to store object screen coordinates. Screen coordinates are
 * (x, y) points that represent an object's location on the computer screen.
 */
struct s_coord {
    int x;
    int y;
};

/**
 * face struct to store object faces. Faces are triangular regions determined by
 * the vertices v1, v2, and v3 and their corresponding vertex normals vn1, vn2,
 * and vn3.
 */
struct face {
    int v1;
    int v2;
    int v3;
    int vn1;
    int vn2;
    int vn3;
};

/**
 * object struct to store an object's name, vector of vertices, vector of faces,
 * the objects' corresponding (world space) transformations, and
 * material properties.
 */
struct object {
    string name;

    // object's material properties.
    color ambi; // ambient
    color diff; // diffuse
    color spec; // specular
    float shin; // shininess

    vector<vertex> vertex_vector;
    vector<vnorm> vnorm_vector;
    vector<face> face_vector;

    // stores translation, rotation, and scaling transformations.
    vector<Matrix4f> t_transform_vector;

    // stores only rotation and scaling transformations.
    vector<Matrix4f> n_transform_vector;
};

#endif
