// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// script_read.h

#ifndef script_read_h
#define script_read_h

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

#include "quaternion.h"

using namespace std;
using namespace Eigen;

/**
 * Camera struct to store the camera's position, orientation, and perspective
 * in world space.
 */
struct Camera {
    /* Index 0 has the x-coordinate
     * Index 1 has the y-coordinate
     * Index 2 has the z-coordinate
     */
    float pos[3]; // position
    float ori_axis[3]; // orientation_axis

    /* Angle in degrees.
     */
    float ori_angle; // orientation_angle

    float n; // near
    float f; // far
    float l; // left
    float r; // right
    float t; // top
    float b; // bottom
};

/**
 * Keyframe struct to store the frame number of the keyframe and the
 * translation, scaling, and rotation transformations corresponding to the
 * keyframe.
 */
struct Keyframe {
    int frame_number;

    Vector3f translation;
    Vector3f scale;
    quaternion rotation;
};

/**
 * Script_Data struct to store the total number of frames in the animation
 * script and a vector with all the keyframes of the animation.
 */
struct Script_Data {
    int n_frames;

    vector<Keyframe> keyframes;
};

/**
 * Print the script data in a way similar to the way the .script files are
 * organized.
 */
void print_script_data(Script_Data sd);

/**
 * Return a Keyframe struct as read from a script file.
 */
Script_Data read_script_file(const string filename);

#endif
