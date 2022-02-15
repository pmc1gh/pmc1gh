// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// diagnostic.cpp

#include "diagnostic.h"
//#include "shading.h"

using namespace std;

/**
 * Print camera information (position, orientation, and perspective).
 */
void print_camera(const camera &cam) {
    cout << "camera:" << endl;
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
 * Print light information (position, color, and attenuation).
 */
void print_light(const light &lt) {
    cout << "light:" << endl;
    cout << "position" << endl;
    cout << lt.x << " " << lt.y << " " << lt.z << endl;
    cout << "color" << endl;
    cout << lt.r << " " << lt.g << " " << lt.b << endl;
    cout << "attenuation" << endl;
    cout << lt.k << endl;
}

/**
 * Print an object's name, vertices, and faces.
 */
void print_object(const object &obj) {
    cout << "object:" << endl;
    cout << "name: " << obj.name << endl;
    vector<vertex> vertex_vector = obj.vertex_vector;
    vector<vnorm> vnorm_vector = obj.vnorm_vector;
    vector<face> face_vector = obj.face_vector;

    cout << "ambient" << endl;
    cout << obj.ambi.r << " " << obj.ambi.g << " " << obj.ambi.b << endl;
    cout << "diffuse" << endl;
    cout << obj.diff.r << " " << obj.diff.g << " " << obj.diff.b << endl;
    cout << "specular" << endl;
    cout << obj.spec.r << " " << obj.spec.g << " " << obj.spec.b << endl;
    cout << "shininess" << endl;
    cout << obj.shin << endl;

    cout << endl;

    // Print the object vertices (1-indexed).
    for (int i = 1; i < (int) vertex_vector.size(); i++) {
        cout << "v" <<  " " << vertex_vector[i].x <<  " "
             << vertex_vector[i].y <<  " " << vertex_vector[i].z << endl;
    }

    cout << endl;

    for (int i = 1; i < (int) vnorm_vector.size(); i++) {
        cout << "vn" <<  " " << vnorm_vector[i].x <<  " "
             << vnorm_vector[i].y <<  " " << vnorm_vector[i].z << endl;
    }

    cout << endl;

    // Print the object faces.
    for (int i = 0; i < (int) face_vector.size(); i++) {
        cout << "f" << " " << face_vector[i].v1 << "//" << face_vector[i].vn1
             <<  " " << face_vector[i].v2 << "//" << face_vector[i].vn2
             <<  " " << face_vector[i].v3 << "//" << face_vector[i].vn3 << endl;
    }

    cout << endl;
}
