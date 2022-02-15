// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// image_functions.h

#include <vector>

#include "general.h"

using namespace std;

/**
 * Build the image by appending pixels of the given background color to the
 * given 2D image vector.
 */
void fill_image_background(vector<vector<pixel>> &image, int xres, int yres,
                           const pixel background_color);

/**
 * Return a buffer grid for use in the shading algorithms for depth buffering.
 */
vector<vector<float>> get_buffer_grid(int xres, int yres);

/**
 * Print the image of resolution (xres, yres) and with the given maximum
 * pixel color intensity.
 */
void print_image(const vector<vector<pixel>> &image,
                 const int xres, const int yres,
                 const int max_intensity);
