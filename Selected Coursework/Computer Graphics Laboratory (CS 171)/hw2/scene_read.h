// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// scene_read.h

#include "general.h"

using namespace std;

/**
 * Read an object's vertices and faces from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
object read_object_file(const string filename);

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
 * Return a vector of all the lights found in a scene description file.
 *
 * invalid_argument thrown if file cannot be opened.
 */
vector<light> read_lights(const string filename);

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
