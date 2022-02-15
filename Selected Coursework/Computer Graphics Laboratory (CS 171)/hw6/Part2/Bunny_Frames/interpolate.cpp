// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// interpolate.cpp

#include <iostream>
#include <vector>

#include "scene_read.h"

using namespace std;

/**
 * Return the four keyframe objects used for interpolation given the vector of
 * keyframe objects and the current frame number.
 */
vector<Object> get_interpolation_keyframe_objects(vector<Object>
                                                        keyframe_objects,
                                                    int current_frame_number) {
    Object kf_before_start, kf_start, kf_end, kf_after_end;
    int i_kf_start, i_kf_end;

    for (int i = 0; i < (int) keyframe_objects.size(); i++) {
        int frame_number = keyframe_objects[i].frame_number;
        if (frame_number <= current_frame_number) {
            kf_start = keyframe_objects[i];
            i_kf_start = i;
        }
        else {
            break;
        }
    }

    if (kf_start.frame_number
        == keyframe_objects[(int) keyframe_objects.size() - 1].frame_number) {
        kf_end = keyframe_objects[0];
        i_kf_end = 0;
    }
    else {
        kf_end = keyframe_objects[i_kf_start+1];
        i_kf_end = i_kf_start+1;
    }

    if (kf_end.frame_number
        == keyframe_objects[(int) keyframe_objects.size() - 1].frame_number) {
        kf_after_end = keyframe_objects[0];
    }
    else {
        kf_after_end = keyframe_objects[i_kf_end+1];
    }

    if (kf_start.frame_number == keyframe_objects[0].frame_number) {
        kf_before_start = keyframe_objects[(int) keyframe_objects.size() - 1];
    }
    else {
        kf_before_start = keyframe_objects[i_kf_start-1];
    }

    vector<Object> interpolation_keyframe_objects;
    interpolation_keyframe_objects.push_back(kf_before_start);
    interpolation_keyframe_objects.push_back(kf_start);
    interpolation_keyframe_objects.push_back(kf_end);
    interpolation_keyframe_objects.push_back(kf_after_end);

    return interpolation_keyframe_objects;
}

/**
 * Return the basis matrix used for interpolation given tension t. Catmull-Rom
 * splines use t = 0.
 */
Matrix4f get_basis_matrix(float t) {
    float s = 0.5 * (1 - t);
    Matrix4f B;
    B << 0, 1, 0, 0,
         -s, 0, s, 0,
         2 * s, s - 3, 3 - 2 * s, -s,
         -s, 2 - s, s - 2, s;
    return B;
}

/**
 * Interpolate a vertex of the current frame given the vector of keyframe
 * objects used for interpolation, the basis matrix B, the * index idx of the
 * vertex in the object's vector of vertices, the current frame * number, the
 * total number of frames in the animation n_frames, and number of frames
 * starting from one keyframe to the next keyframe in the animation
 * keyframe_step.
 */
Vertex get_interpolated_vertex(vector<Object> interpolation_keyframe_objects,
                        Matrix4f B, int idx, int current_frame_number,
                        int n_frames, int keyframe_step) {
    float u;
    if (interpolation_keyframe_objects[2].frame_number
        > interpolation_keyframe_objects[1].frame_number) {
        u = (float) (current_frame_number - interpolation_keyframe_objects[1].frame_number)
            / (interpolation_keyframe_objects[2].frame_number
               - interpolation_keyframe_objects[1].frame_number);
    }
    else {
        u = 0;
    }

    Vector4f u_vector;
    u_vector << 1, u, u * u, u * u * u;

    Vector4f x_values;
    Vector4f y_values;
    Vector4f z_values;

    /* Use the second interpolation keyframe object as the first and second
     * interpolation keyframe objects in interpolation if the current frame
     * number is less than the frame number of the second keyframe in the
     * animation.
     */
    if (current_frame_number < n_frames / keyframe_step) {
        x_values << interpolation_keyframe_objects[1].vertices[idx].x,
                    interpolation_keyframe_objects[1].vertices[idx].x,
                    interpolation_keyframe_objects[2].vertices[idx].x,
                    interpolation_keyframe_objects[3].vertices[idx].x;

        y_values << interpolation_keyframe_objects[1].vertices[idx].y,
                    interpolation_keyframe_objects[1].vertices[idx].y,
                    interpolation_keyframe_objects[2].vertices[idx].y,
                    interpolation_keyframe_objects[3].vertices[idx].y;

        z_values << interpolation_keyframe_objects[1].vertices[idx].z,
                    interpolation_keyframe_objects[1].vertices[idx].z,
                    interpolation_keyframe_objects[2].vertices[idx].z,
                    interpolation_keyframe_objects[3].vertices[idx].z;
    }

    /* Use the third interpolation keyframe object as the third and fourth
     * interpolation keyframe objects in interpolation if the current frame
     * number is greater than the frame number of the second-to-last keyframe in the
     * animation (since the last keyframe in the animation is also the last
     * frame in the animation).
     */
    else if (current_frame_number > n_frames - keyframe_step) {
        x_values << interpolation_keyframe_objects[0].vertices[idx].x,
                    interpolation_keyframe_objects[1].vertices[idx].x,
                    interpolation_keyframe_objects[2].vertices[idx].x,
                    interpolation_keyframe_objects[2].vertices[idx].x;

        y_values << interpolation_keyframe_objects[0].vertices[idx].y,
                    interpolation_keyframe_objects[1].vertices[idx].y,
                    interpolation_keyframe_objects[2].vertices[idx].y,
                    interpolation_keyframe_objects[2].vertices[idx].y;

        z_values << interpolation_keyframe_objects[0].vertices[idx].z,
                    interpolation_keyframe_objects[1].vertices[idx].z,
                    interpolation_keyframe_objects[2].vertices[idx].z,
                    interpolation_keyframe_objects[2].vertices[idx].z;
    }

    else {
        x_values << interpolation_keyframe_objects[0].vertices[idx].x,
                    interpolation_keyframe_objects[1].vertices[idx].x,
                    interpolation_keyframe_objects[2].vertices[idx].x,
                    interpolation_keyframe_objects[3].vertices[idx].x;

        y_values << interpolation_keyframe_objects[0].vertices[idx].y,
                    interpolation_keyframe_objects[1].vertices[idx].y,
                    interpolation_keyframe_objects[2].vertices[idx].y,
                    interpolation_keyframe_objects[3].vertices[idx].y;

        z_values << interpolation_keyframe_objects[0].vertices[idx].z,
                    interpolation_keyframe_objects[1].vertices[idx].z,
                    interpolation_keyframe_objects[2].vertices[idx].z,
                    interpolation_keyframe_objects[3].vertices[idx].z;
    }

    float interpolated_x = u_vector.dot(B * x_values);
    float interpolated_y = u_vector.dot(B * y_values);
    float interpolated_z = u_vector.dot(B * z_values);

    Vertex interpolated_v;
    interpolated_v.x = interpolated_x;
    interpolated_v.y = interpolated_y;
    interpolated_v.z = interpolated_z;

    return interpolated_v;
}

/**
 * Print the given object's vertices and faces to a file of the given file name.
 */
void print_obj_file(Object obj, string filename) {
    ofstream ofs;
    ofs.open(filename);
    // Make sure the file was opened successfully
    if (!ofs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    for (int i = 1; i  < (int) obj.vertices.size(); i++) {
        ofs << "v " << obj.vertices[i].x << " " << obj.vertices[i].y << " "
            << obj.vertices[i].z << endl;
    }

    for (int i = 0; i  < (int) obj.faces.size(); i++) {
        ofs << "f " << obj.faces[i].idx1 << " " << obj.faces[i].idx2 << " "
            << obj.faces[i].idx3 << endl;
    }

    ofs.close();
}

/* The 'main' function:
 *
 * Read the keyframe objects from the keyframes directory and output
 * the 16 interpolated frames for the full animation from frame 0 to 20.
 */
int main(int argc, char* argv[]) {
    int start_frame = 0;
    int end_frame = 20;
    int keyframe_step = 5;
    vector<Object> keyframe_objects;

    // Create the vector of keyframe objects.
    for (int i = start_frame; i <= end_frame; i+=keyframe_step) {
        Object new_object;
        if (i < 10) {
            new_object = read_object_file((string) "keyframes/bunny0"
                                          + to_string(i) + ".obj");
        }
        else {
            new_object = read_object_file((string) "keyframes/bunny"
                                          + to_string(i) + ".obj");
        }
        new_object.frame_number = i;
        keyframe_objects.push_back(new_object);
    }

    vector<Face> faces = keyframe_objects[0].faces;
    Matrix4f B = get_basis_matrix(0);
    for (int frame_number = start_frame; frame_number <= end_frame;
         frame_number++) {
        // Initiliaze the interpolated object.
        Object interpolated_object;
        interpolated_object.frame_number = frame_number;
        interpolated_object.faces = faces;

        /* Interpolate the frame object if the current frame number does not
         * correspond to a keyframe in the animation.
         */
        if (frame_number % keyframe_step != 0) {
            // Create the interpolated vector of vertices.
            vector<Vertex> vertices;
            Vertex null_vertex;
            null_vertex.x = 0;
            null_vertex.y = 0;
            null_vertex.z = 0;
            vertices.push_back(null_vertex);

            vector<Object> interpolation_keyframe_objects =
                get_interpolation_keyframe_objects(keyframe_objects, frame_number);
            for (int idx = 1; idx < (int) keyframe_objects[0].vertices.size();
                idx++) {
                vertices.push_back(get_interpolated_vertex(
                                   interpolation_keyframe_objects, B, idx,
                                   frame_number, end_frame, keyframe_step));
            }
            interpolated_object.vertices = vertices;
            cout << "Frame " << frame_number << " interpolated" << endl;
        }

        /* If the current frame number corresponds to a keyframe in the
         * animation, use the respective keyframe object's vertices for the
         * current frame object.
         */
        else {
            interpolated_object.vertices =
                keyframe_objects[(int) frame_number / keyframe_step].vertices;
        }

        // Print the object to a .obj file.
        string output_filename;
        if (frame_number < 10) {
            output_filename = (string) "output/bunny0" + to_string(frame_number)
                                     + ".obj";
        }
        else {
            output_filename = (string) "output/bunny" + to_string(frame_number)
                                     + ".obj";
        }
        print_obj_file(interpolated_object, output_filename);
    }
}
