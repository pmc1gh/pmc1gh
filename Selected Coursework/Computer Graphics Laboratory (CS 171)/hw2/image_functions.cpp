// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// image_functions.cpp

#include "image_functions.h"

using namespace std;

/**
 * Build the image by appending pixels of the given background color to the
 * given 2D image vector.
 */
void fill_image_background(vector<vector<pixel>> &image, int xres, int yres,
                           const pixel background_color) {
    for (int y = 0; y < yres; y++) {
        vector<pixel> row;
        for (int x = 0; x < xres; x++) {
            row.push_back(background_color);
        }
        image.push_back(row);
    }
}

/**
 * Return a buffer grid for use in the shading algorithms for depth buffering.
 */
vector<vector<float>> get_buffer_grid(int xres, int yres) {
    float max_value = (int) pow(10, 5);
    vector<vector<float>> buffer_grid;
    for (int y = 0; y < yres; y++) {
        vector<float> row;
        for (int x = 0; x < xres; x++) {
            row.push_back(max_value);
        }
        buffer_grid.push_back(row);
    }
    return buffer_grid;
}

/**
 * Print the image of resolution (xres, yres) and with the given maximum
 * pixel color intensity.
 */
void print_image(const vector<vector<pixel>> &image,
                 const int xres, const int yres,
                 const int max_intensity) {
    cout << "P3" << "\n" << xres << " " << yres << "\n" << max_intensity
         << endl;

    // Print the image pixels.
    for (int y = 0; y < (int) image.size(); y++) {
        for (int x = 0; x < (int) image[0].size(); x++) {
            cout << image[y][x].r << " " << image[y][x].g << " "
                 << image[y][x].b << endl;
        }
    }
}
