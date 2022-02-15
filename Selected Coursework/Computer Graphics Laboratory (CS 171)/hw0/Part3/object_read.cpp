// Philip Carr
// CS 171 Assignment 0 Part 3
// October 10, 2018
// object_read.cpp

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>

#include "object.h"

using namespace std;
using namespace Eigen;

/**
 * Read an object's vertices and faces from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
object read_object_file(const string filename) {
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
 * Return true if a given string is found in a vector of strings. Otherwise
 * return false.
 */
bool string_in_vector(const vector<string> &v, const string a) {
    for (int i = 0; i < (int) v.size(); i++) {
        if (v[i] == a) {
            return true;
        }
    }
    return false;
}

/**
 * Return the index of a given string in a vector of strings if the given string
 * is found. Otherwise, return -1 if the given string is not found in the
 * vector.
 */
int string_index_in_vector(const vector<string> &v, const string a) {
    for (int i = 0; i < (int) v.size(); i++) {
        if (v[i] == a) {
            return i;
        }
    }
    return -1;
}

/**
 * Return a vector of the transformation of a vector of vertices by the
 * transformations in a vector of transformations.
 */
vector<vertex> get_transformed_vertices(vector<vertex> v_vector,
                                        vector<Matrix4d> m_vector) {
    Matrix4d transform;
    transform << 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

    // Multiply all the transformations together into one transformation
    for (int i = 0; i < (int) m_vector.size(); i++) {
        transform = m_vector[i] * transform;
    }

    vector<vertex> tv_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    vertex null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    tv_vector.push_back(null_vertex);

    // Transform all the vertices in the given vector of vertices
    for (int i = 1; i < (int) v_vector.size(); i++) {
        Vector4d v;
        v << v_vector[i].x, v_vector[i].y, v_vector[i].z, 1.0;
        Vector4d tv;
        tv = transform * v;
        vertex t_vertex;
        t_vertex.x = tv(0,0);
        t_vertex.y = tv(1,0);
        t_vertex.z = tv(2,0);
        tv_vector.push_back(t_vertex);
    }

    return tv_vector;
}

/**
 * Print an object's name, vertices, and faces.
 */
void print_object(const string name, const object &object) {
    cout << name << ":\n" << endl;
    vector<face> face_vector = object.face_vector;
    vector<vertex> t_vertex_vector;
    t_vertex_vector = get_transformed_vertices(object.vertex_vector,
                                               object.transform_vector);

    // Print the object vertices (1-indexed)
    for (int i = 1; i < (int) t_vertex_vector.size(); i++) {
        cout << "v" <<  " " << t_vertex_vector[i].x <<  " "
             << t_vertex_vector[i].y <<  " " << t_vertex_vector[i].z << endl;
    }

    // Print the object faces
    for (int i = 0; i < (int) face_vector.size(); i++) {
        cout << "f" << " " << face_vector[i].v1 <<  " " << face_vector[i].v2
             <<  " " << face_vector[i].v3 << endl;
    }

    cout << endl;
}

/**
 * Read all the objects (vertices, faces, and corresponding transformations)
 * from a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if neither "t", "r", "s", nor an object name is
 * found at beginning of a non-empty line.
 */
vector<object> read_objects(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename);

    // Make sure the file was opened successfully
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Declare phase to keep track of which part of file is being read (phase
     * of 0 indicates object files are being read, phase of 1 indicates that
     * corresponding object transformations are being read)
     */
    int phase = 0;

    /* Declare index to keep track of which object corresponds to the object
     * transformations currently being read
     */
    int index = -1;

    vector<string> object_names;
    vector<object> base_objects;
    vector<object> objects;
    vector<int> name_occurances;
    int last_obj_index = (int) objects.size() - 1;

    while (true) {
        if (phase == 0) {
            string name;
            ifs >> name;

            // Check if done reading file
            if (ifs.eof()) {
                /* Return the vector of objects if no corresponding object
                 * transformations are present in the file
                 */
                return base_objects;
            }

            /* Read object's corresponding transformations if object name found
             * for a second time
             */
            if (string_in_vector(object_names, name)) {
                index = string_index_in_vector(object_names, name);
                objects.push_back(base_objects[index]);
                name_occurances[index]++;
                phase = 1;
            }

            /* Append object name to vector of object names and append object as
             * read from the corresponding object file (containing object's
             * vertices and faces) to vector of base objects (objects as read
             * from an object file with no corresponding transformations)
             */
            else {
                object_names.push_back(name);
                name_occurances.push_back(0);
                string filename;
                ifs >> filename;
                base_objects.push_back(read_object_file(filename));
            }
        }
        else {
            last_obj_index = (int) objects.size() - 1;
            string d_type;
            ifs >> d_type;

            // Check if done reading file
            if (ifs.eof()) {
                break;
            }

            // Create translation matrix if "t" found
            if (d_type == "t") {
                double tx;
                double ty;
                double tz;
                ifs >> tx >> ty >> tz;
                Matrix4d new_matrix;
                new_matrix << 1, 0, 0, tx,
                              0, 1, 0, ty,
                              0, 0, 1, tz,
                              0, 0, 0, 1;
                objects[last_obj_index].transform_vector.push_back(new_matrix);
            }

            // Create rotation matrix if "r" found
            else if (d_type == "r") {
                double rx;
                double ry;
                double rz;
                double angle;
                ifs >> rx >> ry >> rz >> angle;
                double a00 = rx * rx + (1 - rx * rx) * cos(angle);
                double a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
                double a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
                double a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
                double a11 = ry * ry + (1 - ry * ry) * cos(angle);
                double a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
                double a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
                double a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
                double a22 = rz * rz + (1 - rz * rz) * cos(angle);
                Matrix4d new_matrix;
                new_matrix << a00, a01, a02, 0,
                              a10, a11, a12, 0,
                              a20, a21, a22, 0,
                              0, 0, 0, 1;
                objects[last_obj_index].transform_vector.push_back(new_matrix);
            }

            // Create scaling matrix if "s" found
            else if (d_type == "s") {
                double sx;
                double sy;
                double sz;
                ifs >> sx >> sy >> sz;
                Matrix4d new_matrix;
                new_matrix << sx, 0, 0, 0,
                              0, sy, 0, 0,
                              0, 0, sz, 0,
                              0, 0, 0, 3;
                objects[last_obj_index].transform_vector.push_back(new_matrix);
            }

            /* Print the current object and then append the current object to
             * the vector of objects (with their corresponding transformations),
             * and then read the next object's corresponding transformations
             */
            else if(string_in_vector(object_names, d_type)) {
                print_object((string) object_names[index] + "_copy"
                             + to_string(name_occurances[index]),
                             objects[last_obj_index]);
                index = string_index_in_vector(object_names, d_type);
                objects.push_back(base_objects[index]);
                name_occurances[index]++;
            }

            /* Throw invalid_argument if neither "t", "r", "s", nor an object
             * name is found at beginning of a non-empty line
             */
            else {
                string error_message = (string) "txt file data type is neither "
                                       + "vertex nor value";
                string error_value = (string) d_type;
                throw invalid_argument(error_message + " (" + error_value + ")");
            }
        }
    }

    // Close the file
    ifs.close();

    // Print the last object in the file.
    print_object((string) object_names[index] + "_copy"
                 + to_string(name_occurances[index]),
                 objects[last_obj_index]);

    return objects;
}

int main(int argc, char *argv []) {
    // Check to make sure there is exactly one command line argument
    if (argc != 2) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " testfileN.txt";
        cout << usage_statement << endl;
        return 1;
    }
    vector<object> objects = read_objects(argv[1]);
    return 0;
}
