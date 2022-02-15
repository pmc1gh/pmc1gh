// Philip Carr
// CS 171 HW 0 Part 1
// October 10, 2018
// obj_read.cpp

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include "object.h"

using namespace std;

/**
 * Read an object's vertices and faces from a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
object read_object(const string filename) {
    ifstream ifs;

    // Open the file to read the object.
    ifs.open(filename);

    // Make sure the file was opened successfully
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    vector<vertex> vertex_vector;
    vector<face> face_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    vertex null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    vertex_vector.push_back(null_vertex);

    while (true) {
        string d_type;
        double a;
        double b;
        double c;
        ifs >> d_type >> a >> b >> c;

        // Check if done reading file
        if (ifs.eof()) {
            break;
        }

        // Add new vertex to vertex vector if "v" found at beginning of line
        if (d_type == "v") {
            vertex new_vertex;
            new_vertex.x = a;
            new_vertex.y = b;
            new_vertex.z = c;
            vertex_vector.push_back(new_vertex);
        }

        // Add new face to face vector if "f" found at beginning of line
        else if (d_type == "f") {
            face new_face;
            new_face.v1 = (int) a;
            new_face.v2 = (int) b;
            new_face.v3 = (int) c;
            face_vector.push_back(new_face);
        }

        // Throw error if neither "v" nor "f" is found at beginning of line
        else {
            string error_message = (string) "obj file data type is neither "
                                   + "vertex nor value";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file
    ifs.close();

    object new_object;
    new_object.vertex_vector = vertex_vector;
    new_object.face_vector = face_vector;
    return new_object;
}

/**
 * Print an object's name (given by filename), vertices, and faces.
 */
void print_object(const string filename, const object &object) {
    cout << filename << ":\n" << endl;
    vector<vertex> vertex_vector = object.vertex_vector;
    vector<face> face_vector = object.face_vector;

    // Print the object vertices (1-indexed)
    for (int i = 1; i < (int) vertex_vector.size(); i++) {
        cout << "v" <<  " " << vertex_vector[i].x <<  " " << vertex_vector[i].y
             <<  " " << vertex_vector[i].z << endl;
    }

    // Print the object faces
    for (int i = 0; i < (int) face_vector.size(); i++) {
        cout << "f" << " " << face_vector[i].v1 <<  " " << face_vector[i].v2
             <<  " " << face_vector[i].v3 << endl;
    }

    cout << endl;
}

int main(int argc, char *argv []) {
    // Check to make sure there is at least one command line argument
    if (argc < 2) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " file1.obj file2.obj ... obj_fileN.obj ";
        cout << usage_statement << endl;
        return 1;
    }

    vector<object> object_vector;

    // Read all the objects' vertices
    for(int i = 1; i < argc; i++) {
        object_vector.push_back(read_object(argv[i]));
    }

    // Read all the objects' faces
    for(int i = 0; i < argc - 1; i++) {
        print_object(argv[i+1], object_vector[i]);
    }

    return 0;
}
