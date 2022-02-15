// Philip Carr
// CS 171 Assignment 1
// October 18, 2018
// scene_read.h

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>

#include "camera.h"
#include "object.h"

using namespace std;
using namespace Eigen;

/**
 * Read an object's vertices and faces from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
object read_object_file(const string filename);

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
 * Read the camera information from a file, starting at the "camera:" token.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "camera:" token is found.
 *
 * invalid_argument thrown if unknown camera information (not related to
 * position, orientation, or perspective) is found.
 */
camera read_camera(const string filename);

/**
 * Print camera information (position, orientation, and perspective).
 */
void print_camera(const camera &cam);

/**
 * Return the transformation that transforms vectors from world space to camera
 * space. This transformation is represented by the matrix (TR)^-1, where R is
 * the rotation matrix that rotates vectors to the orientation specified by the
 * camera, and T is the translation matrix that translates vectors specified
 * by the camera's position.
 */
Matrix4f get_world_to_camera_transformation(position pos, orientation ori);

/**
 * Return the transformation that transforms vectors from camera space to
 * homogeneous normalized coordinate space (homogeneous NDC space). This
 * transformation is represented by the matrix (TR)^-1, where R is the
 * perspective_projection matrix that transforms vectors as specified by the
 * camera's perspective.
 */
Matrix4f get_perspective_projection_matrix(perspective per);

/**
 * Return a vector containing vertices transformed by an object's vector of
 * transformations, a world space to camera space transformation, a perspective
 * projection transformation, and division by the w coordinate (since the
 * perspective projection transformation might not keep w = 1, requiring the
 * transformed vector to be normalized again to bring w = 1 again). The vector
 * of vertices returned from this function are Cartesian NDC.
 */
vector<vertex> get_transformed_vertices(const vector<vertex> &v_vector,
                                        const vector<Matrix4f> &m_vector,
                                        const camera &camera);

/**
 * Print an object's name, vertices (transformed into vertices of Cartesian
 * NDC), and faces.
 */
void print_object(const string name, const object &object,
                  const camera &camera);

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
vector<object> read_objects(const string filename);

/**
 * Return a vector of vertices transformed from Cartesian NDC to screen
 * coordinates (s_vertices) (x, y, z), where x and y are integers corresponding
 * to pixel locations in an image. Screen vertices contain another int, v_num,
 * corresponding to the vertex number of the object containing the vertex.
 */
vector<s_vertex> get_screen_vertices(const vector<vertex>
                                       &ndc_cartesian_vertices,
                                   const int xres, const int yres);
