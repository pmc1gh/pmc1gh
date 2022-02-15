// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// scene_read.cpp

#include "scene_read.h"

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

    string filename2;
    if (filename.substr(0,5) != "data/") {
        filename2 = (string) "data/" + filename;
    }
    else {
        filename2 = filename;
    }
    // Open the file to read the object.
    ifs.open(filename2);

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    vector<vertex> vertex_vector;
    vector<vnorm> vnorm_vector;
    vector<face> face_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    vertex null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    vertex_vector.push_back(null_vertex);

    /* Append a "null" vnorm as the 0th element of the vnorm vector so other
     * vnorms can be 1-indexed.
     */
    vnorm null_vnorm;
    null_vnorm.x = 0;
    null_vnorm.y = 0;
    null_vnorm.z = 0;
    vnorm_vector.push_back(null_vnorm);

    while (true) {
        string d_type;
        float a;
        float b;
        float c;
        float d;
        float e;
        float f;
        char s1;
        char s2;
        char s3;
        char s4;
        char s5;
        char s6;
        ifs >> d_type;

        // Check if done reading file.
        if (ifs.eof()) {
            break;
        }

        // Add new vertex to vertex vector if "v" found at beginning of line.
        if (d_type == "v") {
            vertex new_vertex;
            ifs >> a >> b >> c;
            new_vertex.x = a;
            new_vertex.y = b;
            new_vertex.z = c;
            vertex_vector.push_back(new_vertex);
        }

        /* Add new vertex to vertex normal vector if "v" found at beginning of
         * line.
         */
        else if (d_type == "vn") {
            vnorm new_vnorm;
            ifs >> a >> b >> c;
            new_vnorm.x = a;
            new_vnorm.y = b;
            new_vnorm.z = c;
            vnorm_vector.push_back(new_vnorm);
        }

        // Add new face to face vector if "f" found at beginning of line.
        else if (d_type == "f") {
            face new_face;
            ifs >> a >> s1 >> s2 >> b >> c >> s3 >> s4 >> d >> e >> s5 >> s6
                >> f;
            new_face.v1 = (int) a;
            new_face.vn1 = (int) b;
            new_face.v2 = (int) c;
            new_face.vn2 = (int) d;
            new_face.v3 = (int) e;
            new_face.vn3 = (int) f;
            face_vector.push_back(new_face);
        }

        /* Throw error if neither "v", "vn" nor "f" is found at beginning of
         * line.
         */
        else {
            string error_message = (string) "obj file data type is neither "
                                   + "vertex, vnorm, face";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file.
    ifs.close();

    object new_object;
    new_object.vertex_vector = vertex_vector;
    new_object.vnorm_vector = vnorm_vector;
    new_object.face_vector = face_vector;
    return new_object;
}

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
camera read_camera(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename);

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "camera:" token is being searched for, 1 indicates camera
     * information (position, orientation, and perspective) are being read.
     */
    int phase = 0;

    camera cam;

    while (true) {
        if (phase == 0) {
            string token;
            ifs >> token;
            if (ifs.eof()) {
                // Throw an error if there is no "camera:" token in the file.
                string error_message = (string) "File " + filename
                                       + "contains no \"camera:\" token.";
                throw invalid_argument(error_message);
            }
            else {
                /* If the "camera:" token is found, set phase = 1 and start
                 * reading in camera information.
                 */
                if (token == "camera:") {
                    phase = 1;
                }
            }
        }
        else {
            string name;

            /* Camera information should always be 8 lines long (1 line for
             * position information, 1 line for orientation information, and 6
             * lines for perspective information).
             */
            int camera_info_lines = 8;

            float a;
            float b;
            float c;
            float d;

            for (int i = 0; i < camera_info_lines; i++) {
                ifs >> name;
                if (name == "position") {
                    ifs >> a >> b >> c;
                    cam.pos.x = a;
                    cam.pos.y = b;
                    cam.pos.z = c;
                }
                else if (name == "orientation") {
                    ifs >> a >> b >> c >> d;
                    cam.ori.rx = a;
                    cam.ori.ry = b;
                    cam.ori.rz = c;
                    cam.ori.angle = d;
                }
                else if (name == "near") {
                    ifs >> a;
                    cam.per.n = a;
                }
                else if (name == "far") {
                    ifs >> a;
                    cam.per.f = a;
                }
                else if (name == "left") {
                    ifs >> a;
                    cam.per.l = a;
                }
                else if (name == "right") {
                    ifs >> a;
                    cam.per.r = a;
                }
                else if (name == "top") {
                    ifs >> a;
                    cam.per.t = a;
                }
                else if (name == "bottom") {
                    ifs >> a;
                    cam.per.b = a;
                }
                else {
                    string error_message = (string) "unknown camera "
                                           + "information found: " + name;
                    throw invalid_argument(error_message);
                }
            }
            break;
        }
    }

    // Close the file.
    ifs.close();

    return cam;
}

/**
 * Return a vector of all the lights found in a scene description file.
 *
 * invalid_argument thrown if file cannot be opened.
 */
vector<light> read_lights(const string filename) { // FIX COMMENTS
    ifstream ifs;

    // Open the file to read the lights.
    ifs.open(filename);

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "light" token is being searched for, 1 indicates light
     * information (position, color, and attenuation parameter) are being read.
     */
    int phase = 0;

    vector<light> light_vector;

    while (true) {
        if (phase == 0) {
            string token;
            ifs >> token;
            if (ifs.eof()) {
                break;
            }
            else {
                /* If the "light" token is found, set phase = 1 and start
                 * reading in light information.
                 */
                if (token == "light") {
                    phase = 1;
                }
            }
        }
        else {
            light new_light;

            float a;
            float b;
            float c;
            string s1;
            float d;
            float e;
            float f;
            string s2;
            float g;

            ifs >> a >> b >> c >> s1 >> d >> e >> f >> s2 >> g;

            new_light.x = a;
            new_light.y = b;
            new_light.z = c;
            new_light.r = d;
            new_light.g = e;
            new_light.b = f;
            new_light.k = g;

            light_vector.push_back(new_light);

            phase = 0;
        }
    }

    // Close the file.
    ifs.close();

    return light_vector;
}

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
vector<object> read_objects(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename);

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "objects:" token is being searched for, 1 indicates object
     * files are being read, phase of 2 indicates that corresponding object
     * transformations are being read).
     */
    int phase = 0;

    /* Declare index to keep track of which object corresponds to the object
     * transformations currently being read.
     */
    int index = -1;

    vector<string> object_names;
    vector<object> base_objects;
    vector<object> objects;
    vector<int> name_occurances;
    int last_obj_index = (int) objects.size() - 1;

    while (true) {
        if (phase == 0) {
            string token;
            ifs >> token;
            if (ifs.eof()) {
                // Throw an error if there is no "objects:" token in the file.
                string error_message = (string) "File " + filename
                                       + "contains no \"objects:\" token.";
                throw invalid_argument(error_message);
            }
            else {
                /* If the "objects:" token is found, set phase = 1 and start
                 * reading in objects.
                 */
                if (token == "objects:") {
                    phase = 1;
                }
            }
        }
        // phase == 1 when the file has been read up to the "objects:" token.
        else if (phase == 1) {
            string name;
            ifs >> name;

            // Check if done reading file.
            if (ifs.eof()) {
                /* Return the vector of objects if no corresponding object
                 * transformations are present in the file.
                 */
                ifs.close();
                return base_objects;
            }

            /* Read object's corresponding transformations if object name found
             * for a second time.
             */
            if (string_in_vector(object_names, name)) {
                index = string_index_in_vector(object_names, name);
                objects.push_back(base_objects[index]);
                name_occurances[index]++;
                phase = 2;
            }

            /* Append object name to vector of object names and append object as
             * read from the corresponding object file (containing object's
             * vertices and faces) to vector of base objects (objects as read
             * from an object file with no corresponding transformations).
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

            // Check if done reading file.
            if (ifs.eof()) {
                break;
            }

            if (d_type == "ambient") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].ambi.r = r;
                objects[last_obj_index].ambi.g = g;
                objects[last_obj_index].ambi.b = b;
            }

            else if (d_type == "diffuse") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].diff.r = r;
                objects[last_obj_index].diff.g = g;
                objects[last_obj_index].diff.b = b;
            }

            else if (d_type == "specular") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].spec.r = r;
                objects[last_obj_index].spec.g = g;
                objects[last_obj_index].spec.b = b;
            }

            else if (d_type == "shininess") {
                float s;
                ifs >> s;
                objects[last_obj_index].shin = s;
            }

            // Create translation matrix if "t" found.
            else if (d_type == "t") {
                float tx;
                float ty;
                float tz;
                ifs >> tx >> ty >> tz;
                Matrix4f new_matrix;
                new_matrix << 1, 0, 0, tx,
                              0, 1, 0, ty,
                              0, 0, 1, tz,
                              0, 0, 0, 1;
                objects[last_obj_index].t_transform_vector.push_back(
                                                               new_matrix);
            }

            // Create rotation matrix if "r" found.
            else if (d_type == "r") {
                float rx;
                float ry;
                float rz;
                float angle;
                ifs >> rx >> ry >> rz >> angle;
                // Normalize the rotation axis
                float rotation_axis_magnitude = rx * rx + ry * ry + rz * rz;
                rx /= rotation_axis_magnitude;
                ry /= rotation_axis_magnitude;
                rz /= rotation_axis_magnitude;
                float a00 = rx * rx + (1 - rx * rx) * cos(angle);
                float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
                float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
                float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
                float a11 = ry * ry + (1 - ry * ry) * cos(angle);
                float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
                float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
                float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
                float a22 = rz * rz + (1 - rz * rz) * cos(angle);
                Matrix4f new_matrix;
                new_matrix << a00, a01, a02, 0,
                              a10, a11, a12, 0,
                              a20, a21, a22, 0,
                              0, 0, 0, 1;
                objects[last_obj_index].t_transform_vector.push_back(
                                                               new_matrix);
                objects[last_obj_index].n_transform_vector.push_back(
                                                               new_matrix);
            }

            // Create scaling matrix if "s" found.
            else if (d_type == "s") {
                float sx;
                float sy;
                float sz;
                ifs >> sx >> sy >> sz;
                Matrix4f new_matrix;
                new_matrix << sx, 0, 0, 0,
                              0, sy, 0, 0,
                              0, 0, sz, 0,
                              0, 0, 0, 1;
                objects[last_obj_index].t_transform_vector.push_back(
                                                               new_matrix);
                objects[last_obj_index].n_transform_vector.push_back(
                                                               new_matrix);
            }

            /* Print the current object and then append the current object to
             * the vector of objects (with their corresponding transformations),
             * and then read the next object's corresponding transformations.
             */
            else if(string_in_vector(object_names, d_type)) {
                string obj_name = (string) object_names[index] + "_copy"
                                  + to_string(name_occurances[index]);
                objects[last_obj_index].name = obj_name;
                index = string_index_in_vector(object_names, d_type);
                objects.push_back(base_objects[index]);
                name_occurances[index]++;
            }

            /* Throw invalid_argument if neither "t", "r", "s", nor an object
             * name is found at beginning of a non-empty line.
             */
            else {
                string error_message = (string) "txt file data type is neither "
                                       + "vertex nor value";
                string error_value = (string) d_type;
                throw invalid_argument(error_message + " (" + error_value + ")");
            }
        }
    }

    string obj_name = (string) object_names[index] + "_copy"
                      + to_string(name_occurances[index]);
    objects[last_obj_index].name = obj_name;

    // Close the file.
    ifs.close();

    return objects;
}
