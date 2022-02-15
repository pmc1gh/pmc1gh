// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// pixel.h

#ifndef pixel_h
#define pixel_h

/**
 * color struct to store floats corresponding to the colors red (r), green (g)
 * and blue (b).
 */
struct color {
    float r;
    float g;
    float b;
};

/**
 * pixel struct to store integers corresponding to the colors red (r), green (g)
 * and blue (b).
 */
struct pixel {
    int r;
    int g;
    int b;
};

#endif
