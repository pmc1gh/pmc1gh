// Philip Carr
// CS 171 Assignment 0 Part 4
// October 10, 2018
// circle_image.cpp

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "pixel.h"

using namespace std;

/**
 * Return an image vector containing a circle.
 */
vector<pixel> make_circle(const int xres, const int yres, const int center_x,
                          const int center_y, const pixel background_color,
                          const pixel circle_color) {
    vector<pixel> image;
    int diameter;

    // Set the circle diameter equal to the half of min(xres, yres)
    diameter = min(xres, yres) / 2;

    for (int i = 0; i < xres * yres; i++) {
        /* Get the (x, y) coordiantes corresponding to index i in the image
         * vector.
         */
        int x = i % xres;
        int y = i / xres;

        /* Append a pixel of the circle color if the (x, y) coordinate is
         * within/at the boundary of the circle
         */
        if ((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y)
            <= (diameter / 2) * (diameter / 2)) {
            image.push_back(circle_color);
        }

        /* Append a pixel of the background color if the (x, y) coordinate is
         * outside the circle
         */
        else {
            image.push_back(background_color);
        }
    }
    return image;
}

void print_image_to_file(const string filename, const vector<pixel> &image,
                         const int xres, const int yres,
                         const int max_intensity) {
    ofstream ofs;
    ofs.open(filename);
    // Make sure the file was opened successfully
    if (!ofs.good()) {
        throw invalid_argument("Couldn't open file");
    }
    ofs << "P3" << "\n" << xres << " " << yres << "\n" << max_intensity << endl;

    // Print the image pixels.
    for (int i = 0; i < (int) image.size(); i++){
        ofs << image[i].r << " " << image[i].g << " " << image[i].b << endl;
    }

    ofs.close();
}

int main(int argc, char *argv []) {
    if (argc != 3) {
        string usage_statement = (string) "Usage: " + argv[0] + " xres yres";
        cout << usage_statement << endl;
        return 1;
    }
    int max_intensity = 255;

    int xres = atoi(argv[1]);
    int yres = atoi(argv[2]);

    int center_x = xres / 2;
    int center_y = yres / 2;

    pixel background_color;
    background_color.r = 0;
    background_color.g = 128;
    background_color.b = 255;

    pixel circle_color;
    circle_color.r = 0;
    circle_color.g = 255;
    circle_color.b = 128;

    vector<pixel> image = make_circle(xres, yres, center_x, center_y,
                                      background_color, circle_color);

    string filename = "circle.ppm";
    print_image_to_file(filename, image, xres, yres, max_intensity);

    return 0;
}
