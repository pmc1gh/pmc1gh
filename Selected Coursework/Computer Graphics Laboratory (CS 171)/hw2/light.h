// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// light.h

#ifndef light_h
#define light_h

/**
 * light struct to store point light source information. Light source is defined
 * by its (x, y, z) coordinate in world space, its (r, g, b) color, and its
 * attenuation parameter k.
 */
struct light {
    //location
    float x;
    float y;
    float z;

    // color
    float r;
    float g;
    float b;

    // attenuation parameter k
    float k;
};

#endif
