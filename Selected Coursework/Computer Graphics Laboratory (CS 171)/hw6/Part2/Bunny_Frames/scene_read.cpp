// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// scene_read.cpp

#include "scene_read.h"

using namespace std;
using namespace Eigen;

/**
 * Read an object's vertices and vertex normal vectors from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
Object read_object_file(const string filename) {
    ifstream ifs;

    // Open the file to read the object.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    vector<Vertex> vertex_vector;
    vector<Face> face_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    Vertex null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    vertex_vector.push_back(null_vertex);

    while (true) {
        string d_type;
        float a;
        float b;
        float c;
        ifs >> d_type;

        // Check if done reading file.
        if (ifs.eof()) {
            break;
        }

        // Add new vertex to vertex vector if "v" found at beginning of line.
        if (d_type == "v") {
            Vertex new_vertex;
            ifs >> a >> b >> c;
            new_vertex.x = a;
            new_vertex.y = b;
            new_vertex.z = c;
            vertex_vector.push_back(new_vertex);
        }

        // Add new face to face vector if "f" found at beginning of line.
        else if (d_type == "f") {
            Face new_face;
            ifs >> a >> b >> c;
            new_face.idx1 = (int) a;
            new_face.idx2 = (int) b;
            new_face.idx3 = (int) c;
            face_vector.push_back(new_face);
        }

        /* Throw error if neither "v", "vn" nor "f" is found at beginning of
         * line.
         */
        else {
            string error_message = (string) "obj file data type is neither "
                                   + "vertex nor face";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file.
    ifs.close();

    Object new_object;
    new_object.vertices = vertex_vector;
    new_object.faces = face_vector;

    return new_object;
}
