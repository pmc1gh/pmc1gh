// Philip Carr
// CS 171 Assignment 1
// October 18, 2018
// object.h

#include <vector>
#include <Eigen/Dense>

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
 * s_vertex struct to store object screen vertices. Screen vertices store an
 * int v_num that corresponds to the vector number of an object. Screen vertices
 * are (x, y, z) points that represent an object's location on the computer
 * screen.
 */
struct s_vertex {
    int v_num;
    int x;
    int y;
    float z;
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
 * the objects' corresponding transformations.
 */
struct object {
    vector<vertex> vertex_vector;
    vector<face> face_vector;
    vector<Matrix4f> transform_vector;
};
