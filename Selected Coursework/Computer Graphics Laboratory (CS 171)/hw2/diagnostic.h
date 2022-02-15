// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// diagnostic.h

#include "general.h"

using namespace std;

/**
 * Print camera information (position, orientation, and perspective).
 */
void print_camera(const camera &cam);

/**
 * Print light information (position, color, and attenuation).
 */
void print_light(const light &lt);

/**
 * Print an object's name, vertices, and faces.
 */
void print_object(const object &object);
