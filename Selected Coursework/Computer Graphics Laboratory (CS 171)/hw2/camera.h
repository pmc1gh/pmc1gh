// Philip Carr
// CS 171 Assignment 1
// October 26, 2018
// camera.h

#ifndef camera_h
#define camera_h

#include <vector>

using namespace std;

/**
 * position struct to store a camera's position in world space. The position
 * is an (x, y, z) point.
 */
struct position {
    float x;
    float y;
    float z;
};

/**
 * orientation struct to store a camera's orientation in world space.
 * The orientation stores the rotation axis and angle magnitude of rotation of
 * the camera.
 */
struct orientation {
    float rx;
    float ry;
    float rz;
    float angle;
};

/**
 * perspective struct to store a camera's perspective projection, defined by the
 * given boundaries represented as floats.
 */
struct perspective {
    float n; // near
    float f; // far
    float l; // left
    float r; // right
    float t; // top
    float b; // bottom
};

/**
 * camera struct to store the camera's position, orientation, and perspective
 * in world space.
 */
struct camera {
    position pos;
    orientation ori;
    perspective per;
};

#endif
