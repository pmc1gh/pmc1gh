// Philip Carr
// CS 171 Assignment 1
// October 18, 2018
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
        float a;
        float b;
        float c;
        ifs >> d_type >> a >> b >> c;

        // Check if done reading file.
        if (ifs.eof()) {
            break;
        }

        // Add new vertex to vertex vector if "v" found at beginning of line.
        if (d_type == "v") {
            vertex new_vertex;
            new_vertex.x = a;
            new_vertex.y = b;
            new_vertex.z = c;
            vertex_vector.push_back(new_vertex);
        }

        // Add new face to face vector if "f" found at beginning of line.
        else if (d_type == "f") {
            face new_face;
            new_face.v1 = (int) a;
            new_face.v2 = (int) b;
            new_face.v3 = (int) c;
            face_vector.push_back(new_face);
        }

        // Throw error if neither "v" nor "f" is found at beginning of line.
        else {
            string error_message = (string) "obj file data type is neither "
                                   + "vertex nor value";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file.
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
                                       + "contains no \"objects:\" token.";
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
 * Print camera information (position, orientation, and perspective).
 */
void print_camera(const camera &cam) {
    cout << "position" << endl;
    cout << cam.pos.x << " " << cam.pos.y << " " << cam.pos.z << " " << endl;
    cout << "orientation" << endl;
    cout << cam.ori.rx << " " << cam.ori.ry << " " << cam.ori.rz << " "
         << cam.ori.angle << endl;
    cout << "perspective" << endl;
    cout << cam.per.n << " " << cam.per.f << "\n"
         << cam.per.l << " " << cam.per.r << "\n"
         << cam.per.t << " " << cam.per.b << endl;
}

/**
 * Return the transformation that transforms vectors from world space to camera
 * space. This transformation is represented by the matrix (TR)^-1, where R is
 * the rotation matrix that rotates vectors to the orientation specified by the
 * camera, and T is the translation matrix that translates vectors specified
 * by the camera's position.
 */
Matrix4f get_world_to_camera_transformation(position pos, orientation ori) {
    float rx = ori.rx;
    float ry = ori.ry;
    float rz = ori.rz;
    float angle = ori.angle;
    float a00 = rx * rx + (1 - rx * rx) * cos(angle);
    float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
    float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
    float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
    float a11 = ry * ry + (1 - ry * ry) * cos(angle);
    float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
    float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
    float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
    float a22 = rz * rz + (1 - rz * rz) * cos(angle);
    Matrix4f rotation_matrix;
    rotation_matrix << a00, a01, a02, 0,
                       a10, a11, a12, 0,
                       a20, a21, a22, 0,
                       0, 0, 0, 1;

    Matrix4f translation_matrix;
    translation_matrix << 1, 0, 0, pos.x,
                          0, 1, 0, pos.y,
                          0, 0, 1, pos.z,
                          0, 0, 0, 1;

    Matrix4f product_matrix = translation_matrix * rotation_matrix;
    Matrix4f w_to_c_transformation = product_matrix.inverse();
    return w_to_c_transformation;
}

/**
 * Return the transformation that transforms vectors from camera space to
 * homogeneous normalized coordinate space (homogeneous NDC space). This
 * transformation is represented by the matrix (TR)^-1, where R is the
 * perspective_projection matrix that transforms vectors as specified by the
 * camera's perspective.
 */
Matrix4f get_perspective_projection_matrix(perspective per) {
    Matrix4f perspective_projection;
    float a00 = 2 * per.n / (per.r - per.l);
    float a02 = (per.r + per.l) / (per.r - per.l);
    float a11 = 2 * per.n / (per.t - per.b);
    float a12 = (per.t + per.b) / (per.t - per.b);
    float a22 = -(per.f + per.n) / (per.f - per.n);
    float a23 = -2 * per.f * per.n / (per.f - per.n);
    perspective_projection << a00, 0, a02, 0,
                              0, a11, a12, 0,
                              0, 0, a22, a23,
                              0, 0, -1, 0;
    return perspective_projection;
}

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
                                        const camera &camera) {
    Matrix4f transform;
    transform << 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

    // Multiply all the transformations together into one transformation.
    for (int i = 0; i < (int) m_vector.size(); i++) {
        transform = m_vector[i] * transform;
    }

    Matrix4f camera_space_transform =
        get_world_to_camera_transformation(camera.pos, camera.ori);

    transform = camera_space_transform * transform;

    Matrix4f perspective_transform =
        get_perspective_projection_matrix(camera.per);

    transform = perspective_transform * transform;

    vector<vertex> tv_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    vertex null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    tv_vector.push_back(null_vertex);

    // Transform all the vertices in the given vector of vertices.
    for (int i = 1; i < (int) v_vector.size(); i++) {
        Vector4f v;
        v << v_vector[i].x, v_vector[i].y, v_vector[i].z, 1.0;
        Vector4f tv;
        tv = transform * v;
        vertex t_vertex;
        t_vertex.x = tv(0,0) / tv(3,0);
        t_vertex.y = tv(1,0) / tv(3,0);
        t_vertex.z = tv(2,0) / tv(3,0);
        tv_vector.push_back(t_vertex);
    }

    return tv_vector;
}

/**
 * Print an object's name, vertices (transformed into vertices of Cartesian
 * NDC), and faces.
 */
void print_object(const string name, const object &object,
                  const camera &camera) {
    cout << name << ":\n" << endl;
    vector<face> face_vector = object.face_vector;
    vector<vertex> t_vertex_vector;
    t_vertex_vector = get_transformed_vertices(object.vertex_vector,
                                               object.transform_vector,
                                               camera);

    // Print the object vertices (1-indexed).
    for (int i = 1; i < (int) t_vertex_vector.size(); i++) {
        cout << "v" <<  " " << t_vertex_vector[i].x <<  " "
             << t_vertex_vector[i].y <<  " " << t_vertex_vector[i].z << endl;
    }

    // Print the object faces.
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

            // Create translation matrix if "t" found.
            if (d_type == "t") {
                float tx;
                float ty;
                float tz;
                ifs >> tx >> ty >> tz;
                Matrix4f new_matrix;
                new_matrix << 1, 0, 0, tx,
                              0, 1, 0, ty,
                              0, 0, 1, tz,
                              0, 0, 0, 1;
                objects[last_obj_index].transform_vector.push_back(new_matrix);
            }

            // Create rotation matrix if "r" found.
            else if (d_type == "r") {
                float rx;
                float ry;
                float rz;
                float angle;
                ifs >> rx >> ry >> rz >> angle;
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
                objects[last_obj_index].transform_vector.push_back(new_matrix);
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
                objects[last_obj_index].transform_vector.push_back(new_matrix);
            }

            /* Print the current object and then append the current object to
             * the vector of objects (with their corresponding transformations),
             * and then read the next object's corresponding transformations.
             */
            else if(string_in_vector(object_names, d_type)) {
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

    // Close the file.
    ifs.close();

    return objects;
}

/**
 * Return a vector of vertices transformed from Cartesian NDC to screen
 * coordinates (s_vertices) (x, y, z), where x and y are integers corresponding
 * to pixel locations in an image. Screen vertices contain another int, v_num,
 * corresponding to the vertex number of the object containing the vertex.
 */
vector<s_vertex> get_screen_vertices(const vector<vertex>
                                       &ndc_cartesian_vertices,
                                   const int xres, const int yres) {
    vector<s_vertex> screen_vertices;
    for (int i = 1; i < (int) ndc_cartesian_vertices.size(); i++) {
        float x = ndc_cartesian_vertices[i].x;
        float y = ndc_cartesian_vertices[i].y;
        if (x > -1 && x < 1 && y > -1 && y < 1) {
            s_vertex screen_vertex;
            float ndc_c_x = ndc_cartesian_vertices[i].x;
            float ndc_c_y = ndc_cartesian_vertices[i].y;
            screen_vertex.v_num = i;
            screen_vertex.x = (int) ((ndc_c_x + 1) * ((xres - 1) / 2));
            screen_vertex.y = (int) ((-ndc_c_y + 1) * ((yres - 1) / 2));
            screen_vertices.push_back(screen_vertex);
        }
    }
    return screen_vertices;
}
