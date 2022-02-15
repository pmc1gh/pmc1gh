// Philip Carr
// CS 171 Assignment 4 Part 1
// November 17, 2018
// scene_read.h

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <vector>

using namespace std;

/**
 * Camera struct to store the camera's position, orientation, and perspective
 * in world space.
 */
struct Camera {
    /* Index 0 has the x-coordinate
     * Index 1 has the y-coordinate
     * Index 2 has the z-coordinate
     */
    float pos[3]; // position
    float ori_axis[3]; // orientation_axis

    /* Angle in degrees.
     */
    float ori_angle; // orientation_angle

    float n; // near
    float f; // far
    float l; // left
    float r; // right
    float t; // top
    float b; // bottom
};

/* Point_Light struct for representing a point light source of a scene. The
 * position is represented in homogeneous coordinates rather than
 * the simple Cartesian coordinates normally used, since OpenGL requires a
 * w-coordinate be specified when the positions of the point lights are
 * specified. The Point_Light positions are specified in the set_lights
 * function.
 */
struct Point_Light
{
    /* Index 0 has the x-coordinate
     * Index 1 has the y-coordinate
     * Index 2 has the z-coordinate
     * Index 3 has the w-coordinate
     */
    float pos[4]; // position

    /* Index 0 has the r-component
     * Index 1 has the g-component
     * Index 2 has the b-component
     */
    float color[3];

    /* This is our 'k' factor for attenuation as discussed in the lecture notes
     * and extra credit of Assignment 2.
     */
    float k; // attenuation_k
};

/**
 * Triple struct for storing vertices and vertex normal vectors (w-coordinates
 * of vertices and vertex normal vectors do not need to be stored here since
 * OpenGL handles them).
 */
struct Triple
{
    float x;
    float y;
    float z;
};

/**
 * Transform struct to store a transformation. Each Transform struct contains
 * an array representation of a translation, rotation, or scaling
 * transformation.
 */
struct Transform
{
    /* Type of transformation.
     * "t" = translation, "r" = rotation, "s" = scaling.
     */
    string type;

    /* For each array below,
     * Index 0 has the x-component
     * Index 1 has the y-component
     * Index 2 has the z-component
     */
    float components[3];

    // Angle in degrees.
    float rotation_angle;
};

/**
 * Object struct to store objects to render. Objects contain a name, vectors
 * of vertices and their corresponding vertex normal vectors, a vector of
 * associated transformations, and material properties used by the lighting
 * model.
 */
struct Object
{
    string name;

    /* See the note above and the comments in the 'draw_objects' and
     * 'create_cubes' functions for details about these buffer vectors.
     */
    vector<Triple> vertex_buffer;
    vector<Triple> normal_buffer;

    vector<Transform> transform_sets;

    /* Index 0 has the r-component
     * Index 1 has the g-component
     * Index 2 has the b-component
     */
    float ambient_reflect[3];
    float diffuse_reflect[3];
    float specular_reflect[3];

    float shininess;
};

/**
 * Return true if a given string is found in a vector of strings. Otherwise
 * return false.
 */
bool string_in_vector(const vector<string> &v, const string a);

/**
 * Return the index of a given string in a vector of strings if the given string
 * is found. Otherwise, return -1 if the given string is not found in the
 * vector.
 */
int string_index_in_vector(const vector<string> &v, const string a);

/**
 * Read an object's vertices and faces from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
Object read_object_file(const string filename);

/**
 * Read the camera information from a file, starting at the "camera:" token.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "camera:" token is found.
 *
 * invalid_argument thrown if unknown camera information (not related to
 * position, orientation, or perspective) is found.
 */
Camera read_camera(const string filename);

/**
 * Return a vector of all the lights found in a scene description file.
 *
 * invalid_argument thrown if file cannot be opened.
 */
vector<Point_Light> read_lights(const string filename);

/**
 * Read all the objects (vertices, faces, and corresponding transformations)
 * from a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "objects:" token is found.
 *
 * invalid_argument thrown if neither "t", "r", "s", nor an object name is
 * found at beginning of a non-empty line.
 */
vector<Object> read_objects(const string filename);
