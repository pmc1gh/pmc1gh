// Philip Carr
// CS 171 Assignment 3
// November 2, 2018
// scene_read.cpp

#include "scene_read.h"

using namespace std;

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
 * Read an object's vertices and vertex normal vectors from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
Object read_object_file(const string filename) {
    ifstream ifs;

    string filename2;
    if (filename.substr(0,5) != "data/") {
        filename2 = (string) "data/" + filename;
    }
    else {
        filename2 = filename;
    }
    // Open the file to read the object.
    ifs.open(filename2.c_str()); // why does c_str have to be used here???

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    vector<Triple> vertex_vector;
    vector<Triple> vnorm_vector;
    //vector<face> face_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    Triple null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    vertex_vector.push_back(null_vertex);

    /* Append a "null" vnorm as the 0th element of the vnorm vector so other
     * vnorms can be 1-indexed.
     */
    Triple null_vnorm;
    null_vnorm.x = 0;
    null_vnorm.y = 0;
    null_vnorm.z = 0;
    vnorm_vector.push_back(null_vnorm);

    /* Object vertex buffer
     */
    vector<Triple> vertex_buffer;

    /* Object normal buffer
     */
    vector<Triple> normal_buffer;

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
            Triple new_vertex;
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
            Triple new_vnorm;
            ifs >> a >> b >> c;
            new_vnorm.x = a;
            new_vnorm.y = b;
            new_vnorm.z = c;
            vnorm_vector.push_back(new_vnorm);
        }

        // Add new face to face vector if "f" found at beginning of line.
        else if (d_type == "f") {
            //face new_face;
            ifs >> a >> s1 >> s2 >> b >> c >> s3 >> s4 >> d >> e >> s5 >> s6
                >> f;
            // new_face.v1 = (int) a;
            // new_face.vn1 = (int) b;
            // new_face.v2 = (int) c;
            // new_face.vn2 = (int) d;
            // new_face.v3 = (int) e;
            // new_face.vn3 = (int) f;
            //face_vector.push_back(new_face);
            vertex_buffer.push_back(vertex_vector[(int) a]);
            vertex_buffer.push_back(vertex_vector[(int) c]);
            vertex_buffer.push_back(vertex_vector[(int) e]);

            normal_buffer.push_back(vnorm_vector[(int) b]);
            normal_buffer.push_back(vnorm_vector[(int) d]);
            normal_buffer.push_back(vnorm_vector[(int) f]);
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

    Object new_object;
    new_object.vertex_buffer = vertex_buffer;
    new_object.normal_buffer = normal_buffer;
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
Camera read_camera(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "camera:" token is being searched for, 1 indicates camera
     * information (position, orientation, and perspective) are being read.
     */
    int phase = 0;

    Camera cam;

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
                    cam.pos[0] = a;
                    cam.pos[1] = b;
                    cam.pos[2] = c;
                }
                else if (name == "orientation") {
                    ifs >> a >> b >> c >> d;
                    cam.ori_axis[0] = a;
                    cam.ori_axis[1] = b;
                    cam.ori_axis[2] = c;
                    cam.ori_angle = d;
                }
                else if (name == "near") {
                    ifs >> a;
                    cam.n = a;
                }
                else if (name == "far") {
                    ifs >> a;
                    cam.f = a;
                }
                else if (name == "left") {
                    ifs >> a;
                    cam.l = a;
                }
                else if (name == "right") {
                    ifs >> a;
                    cam.r = a;
                }
                else if (name == "top") {
                    ifs >> a;
                    cam.t = a;
                }
                else if (name == "bottom") {
                    ifs >> a;
                    cam.b = a;
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
vector<Point_Light> read_lights(const string filename) {
    ifstream ifs;

    // Open the file to read the lights.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "light" token is being searched for, 1 indicates light
     * information (position, color, and attenuation parameter) are being read.
     */
    int phase = 0;

    vector<Point_Light> light_vector;

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
            Point_Light new_light;

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

            new_light.pos[0] = a;
            new_light.pos[1] = b;
            new_light.pos[2] = c;
            new_light.pos[3] = 1; // w coordinate
            new_light.color[0] = d;
            new_light.color[1] = e;
            new_light.color[2] = f;
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
 * Read all the objects (vertices, vertex normal vectors, and corresponding
 * transformations) from a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "objects:" token is found.
 *
 * invalid_argument thrown if neither "t", "r", "s", nor an object name is
 * found at beginning of a non-empty line.
 */
vector<Object> read_objects(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename.c_str());

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
    vector<Object> base_objects;
    vector<Object> objects;
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
                objects[last_obj_index].ambient_reflect[0] = r;
                objects[last_obj_index].ambient_reflect[1] = g;
                objects[last_obj_index].ambient_reflect[2] = b;
            }

            else if (d_type == "diffuse") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].diffuse_reflect[0] = r;
                objects[last_obj_index].diffuse_reflect[1] = g;
                objects[last_obj_index].diffuse_reflect[2] = b;
            }

            else if (d_type == "specular") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].specular_reflect[0] = r;
                objects[last_obj_index].specular_reflect[1] = g;
                objects[last_obj_index].specular_reflect[2] = b;
            }

            else if (d_type == "shininess") {
                float s;
                ifs >> s;
                objects[last_obj_index].shininess = s;
            }

            // Create translation matrix if "t" found.
            else if (d_type == "t") {
                float tx;
                float ty;
                float tz;
                ifs >> tx >> ty >> tz;
                Transforms t_transform;
                t_transform.translation[0] = tx;
                t_transform.translation[1] = ty;
                t_transform.translation[2] = tz;
                t_transform.rotation[0] = 1;
                t_transform.rotation[1] = 0;
                t_transform.rotation[2] = 0;
                t_transform.scaling[0] = 1;
                t_transform.scaling[1] = 1;
                t_transform.scaling[2] = 1;
                t_transform.rotation_angle = 0;
                objects[last_obj_index].transform_sets.push_back(t_transform);
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
                Transforms r_transform;
                r_transform.translation[0] = 0;
                r_transform.translation[1] = 0;
                r_transform.translation[2] = 0;
                r_transform.rotation[0] = rx;
                r_transform.rotation[1] = ry;
                r_transform.rotation[2] = rz;
                r_transform.scaling[0] = 1;
                r_transform.scaling[1] = 1;
                r_transform.scaling[2] = 1;
                r_transform.rotation_angle = angle * 180.0 / M_PI;
                objects[last_obj_index].transform_sets.push_back(r_transform);
            }

            // Create scaling matrix if "s" found.
            else if (d_type == "s") {
                float sx;
                float sy;
                float sz;
                ifs >> sx >> sy >> sz;
                Transforms s_transform;
                s_transform.translation[0] = 0;
                s_transform.translation[1] = 0;
                s_transform.translation[2] = 0;
                s_transform.rotation[0] = 1;
                s_transform.rotation[1] = 0;
                s_transform.rotation[2] = 0;
                s_transform.scaling[0] = sx;
                s_transform.scaling[1] = sy;
                s_transform.scaling[2] = sz;
                s_transform.rotation_angle = 0;
                objects[last_obj_index].transform_sets.push_back(s_transform);
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
                throw invalid_argument(error_message
                                       + " (" + error_value + ")");
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
