// Philip Carr
// CS 171 Assignment 1
// October 18, 2018
// wireframe_renderer.cpp

#include "scene_read.h"
#include "pixel.h"

using namespace std;
using namespace Eigen;

/**
 * Return the index of a given s_vertex in a vector of s_vertices if the given
 * s_vertex is found. Otherwise, return -1 if the given s_vertex is not found in
 * the vector.
 */
int s_vertex_index_in(const vector<s_vertex> &sv, const int v_num) {
    for (int i = 0; i < (int) sv.size(); i++) {
        if (sv[i].v_num == v_num) {
            return i;
        }
    }
    return -1;
}

/**
 * Bresenham's line algorithm for drawing lines in an image grid.
 *
 * Explanation of Bresenham's line algorithm (line drawing in the first octant):
 *
 * Line drawing starts at (x0, y0), with this point being marked with the line
 * color. A new point in the line is marked each time x is incremented. y
 * increments at a rate depending on the line slope m.
 *
 * To determine when y should be incremented (by 1), introduce an error term
 * epsilon (represented as e below) that accumilates error in y (as x increases
 * from x0 to x1) by increments of the line slope m. When (e + m) becomes more
 * than 0.5 (pixels), e is incremented by m then decremented by 1 to reset the
 * slope error for the next time y should be incremented, and y is incremented
 * by 1.
 *
 * Line drawing ends at (x1, y1), with this point being marked.
 *
 * Explanation of generalized Bresenham's line algorithm:
 *
 * Generalization to octant 5:
 *
 * Lines in octant 5 have the same slope as in octant 1, but x0 > x1. Swapping
 * (x0, y0) and (x1, y1) and then iterating from x0 to x1 results in drawing a
 * line as if it were originally in octant 1.
 *
 * Generalization to octant 8:
 *
 * Lines in octant 8 have slopes negative of those as in octant 1. Instead of
 * incrementing y by 1 when (e + m) > 0.5, decrementing y by 1 when
 * (e + m) < -0.5 draws a line with a negative slope of the corresponding line
 * in octant 1.
 *
 * Generalization to octant 4:
 *
 * Lines in octant 4 have the same slopes as in octant 8, but x0 > x1. Swapping
 * (x0, y0) and (x1, y1) and then iterating from x0 to x1 results in drawing a
 * line as if it were originally in octant 8.
 *
 * Generalization to octant 2:
 *
 * Lines in octant 2 have slopes greater than 1, so while slope m > 1,
 * (1 / m) < 1. Thus, lines can be drawn with the same process used for octant
 * 1, but with iteration from y0 to y1 instead of iteration from x0 to x1,
 * incrementing x by 1 when e + (1 / m) > 0.5, since (1 / m) is the change in x
 * per unit change in y.
 *
 * Generalization to octant 3:
 *
 * Lines in octant 3 have slopes negative of those as in octant 2. Instead of
 * incrementing x by 1 when e + (1 / m) > 0.5, decrementing x by 1 when
 * e + (1 / m) < -0.5 draws a line with a negative slope of the corresponding
 * line in octant 2.
 *
 * Generalization to octant 6:
 *
 * Lines in octant 6 have the same range of absolute value of slope as in octant
 * 2, but y0 > y1. Swapping (x0, y0) and (x1, y1) and then iterating from y0 to
 * y1 results in drawing a line as if it were originally in octant 2.
 *
 * Generalization to octant 7:
 *
 * Lines in octant 7 have the same slopes as in octant 3, but y0 > y1. Swapping
 * (x0, y0) and (x1, y1) and then iterating from y0 to y1 results in drawing a
 * line as if it were originally in octant 3.
 */
void bresenham(int x_0, int y_0, int x_1, int y_1, float m, int octant,
               vector<vector<pixel>> &image, pixel line_color) {
    float e = 0;
    int x0;
    int y0;
    int x1;
    int y1;

    // Swap (x0, y0) and (x1, y1) if octant is 4, 5, 6, or 7.
    if (octant == 4 || octant == 5 || octant == 6 || octant == 7) {
        x0 = x_1;
        x1 = x_0;
        y0 = y_1;
        y1 = y_0;
    }

    else {
        x0 = x_0;
        x1 = x_1;
        y0 = y_0;
        y1 = y_1;
    }

    /* If |m| < 1, iterate over x and increment or decrement y depending on the
     * line slope m.
     */
    if (octant == 1 || octant == 8 || octant == 4 || octant == 5) {
        int y = y0;
        for (int x = x0; x < x1; x++) {
            image[y][x] = line_color;
            if (octant == 1 || octant == 5) {
                if (e + m < 0.5) {
                    e += m;
                }
                else {
                    e += m - 1;
                    y += 1;
                }
            }
            else {
                if (e + m > -0.5) {
                    e += m;
                }
                else {
                    e += m + 1;
                    y -= 1;
                }
            }
        }
    }

    /* If |m| > 1, iterate over y and increment or decrement x depending on the
     * line slope m.
     */
    else {
        float m_inv = 1 / m;
        int x = x0;
        for (int y = y0; y < y1; y++) {
            image[y][x] = line_color;
            if (octant == 2 || octant == 6) {
                if (e + m_inv < 0.5) {
                    e += m_inv;
                }
                else {
                    e += m_inv - 1;
                    x += 1;
                }
            }
            else {
                if (e + m_inv > -0.5) {
                    e += m_inv;
                }
                else {
                    e += m_inv + 1;
                    x -= 1;
                }
            }
        }
    }
}

/**
 * Draw a line between two s_vertices on the given image with the given line
 * color using a generalized form of Bresenham's line algorithm for drawing
 * lines.
 */
void draw_line(s_vertex sv0, s_vertex sv1, vector<vector<pixel>> &image,
               pixel line_color) {
    int x0 = sv0.x;
    int y0 = sv0.y;
    int x1 = sv1.x;
    int y1 = sv1.y;
    float m = ((float) y1 - y0) / (x1 - x0);
    int octant;

    // Conditions for lines found in the octants of the coordinate plane.
    if (m >= 0 && m <= 1 && x0 < x1) {
        octant = 1;
    }
    else if (m < 0 && m >= -1 && x0 < x1) {
        octant = 8;
    }
    else if (m < 0 && m >= -1 && x0 > x1) {
        octant = 4;
    }
    else if (m >= 0 && m <= 1 && x0 > x1) {
        octant = 5;
    }
    else if (m > 1 && y0 < y1) {
        octant = 2;
    }
    else if (m < -1 && y0 < y1) {
        octant = 3;
    }
    else if (m > 1 && y0 > y1) {
        octant = 6;
    }
    else {
        octant = 7;
    }

    bresenham(x0, y0, x1, y1, m, octant, image, line_color);
}

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

int main(int argc, char *argv []) {
    // Check to make sure there are exactly 3 command line arguments.
    if (argc != 4) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [scene_description_file.txt] [xres]"
                                 + " [yres]\n" + "Add \" | display -\" after"
                                 + " [yres] to directly display the PPM image\n"
                                 + "Add \" | convert - my_image_name.png \""
                                 + " after [yres] to convert the PPM image into"
                                 + " a PNG image.";
        cout << usage_statement << endl;
        return 1;
    }

    int max_intensity = 255;

    int xres = atoi(argv[2]);
    int yres = atoi(argv[3]);

    pixel background_color;
    background_color.r = 0;
    background_color.g = 0;
    background_color.b = 0;

    pixel line_color;
    line_color.r = 0;
    line_color.g = 0;
    line_color.b = 255;

    vector<vector<pixel>> image;
    fill_image_background(image, xres, yres, background_color);

    camera cam = read_camera(argv[1]);

    vector<object> objects = read_objects(argv[1]);

    for (int i = 0; i < (int) objects.size(); i++) {
        vector<vertex> ndc_cartesian_vertices =
            get_transformed_vertices(objects[i].vertex_vector,
                                     objects[i].transform_vector, cam);

        vector<s_vertex> s_vertex_vector =
            get_screen_vertices(ndc_cartesian_vertices, xres, yres);

        for (int j = 0; j < (int) objects[i].face_vector.size(); j++) {
            int v1 = objects[i].face_vector[j].v1;
            int v2 = objects[i].face_vector[j].v2;
            int v3 = objects[i].face_vector[j].v3;
            int v1_index = s_vertex_index_in(s_vertex_vector, v1);
            int v2_index = s_vertex_index_in(s_vertex_vector, v2);
            int v3_index = s_vertex_index_in(s_vertex_vector, v3);

            /* Draw lines between every pair of the three vertices that are both
             * found in the screen coordinates (vector of s_vertices).
             */
            if (v1_index != -1 && v2_index != -1) {
                draw_line(s_vertex_vector[v1_index], s_vertex_vector[v2_index],
                          image, line_color);
            }
            if (v1_index != -1 && v3_index != -1) {
                draw_line(s_vertex_vector[v1_index], s_vertex_vector[v3_index],
                          image, line_color);
            }
            if (v2_index != -1 && v3_index != -1) {
                draw_line(s_vertex_vector[v2_index], s_vertex_vector[v3_index],
                          image, line_color);
            }
        }
    }

    print_image(image, xres, yres, max_intensity);

    return 0;
}
