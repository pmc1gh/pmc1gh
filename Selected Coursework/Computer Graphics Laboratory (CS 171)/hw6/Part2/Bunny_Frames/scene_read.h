// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// scene_read.h

#ifndef scene_read_h
#define scene_read_h

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/**
 * Vertex struct to store vertex data ((x, y, z) position of the vertex).
 */
struct Vertex {
    float x;
    float y;
    float z;
};

/**
 * Face struct to store face data (integers idx1, idx2, and idx3 corresponding
 * to the indices in the corresponding vertex vector of the three vertices of
 * the given face).
 */
struct Face {
    int idx1;
    int idx2;
    int idx3;
};

/**
 * Object struct to store objects to render. Objects contain a name, vectors
 * of vertices and their corresponding vertex normal vectors, a vector of
 * associated transformations, and material properties used by the lighting
 * model.
 */
struct Object {
    int frame_number;
    vector<Vertex> vertices;
    vector<Face> faces;
};

/**
 * Return a Mesh_Data struct of an object's vertices and faces from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
Object read_object_file(const string filename);

#endif
