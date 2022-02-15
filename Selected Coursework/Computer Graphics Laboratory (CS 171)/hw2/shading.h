// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// shading.h

#include "transformation.h"

using namespace std;

/**
 * Return the color of a given vertex with a given vertex normal vector
 * determined by the implemented lighting model.
 */
color lighting(const vertex v, const vnorm vn, const object &obj,
               const vector<light> &lights, const camera &cam);

/**
 * Helper function for computing alpha, beta, and gamma used in triangle
 * barycentric coordinates.
 */
float compute_f_ij(const float xi, const float yi, const float xj,
                   const float yj, const float x, const float y);

/**
 * Return alpha as given by the definition of triangle barycentric coordinates
 * (alpha is the coefficient for triangle vertex a in the equation
 * ndc_point = alpha * ndc_a + beta * ndc_b + gamma * ndc_c).
 */
float compute_alpha(const s_coord &a, const s_coord &b, const s_coord &c,
                    const float x, const float y);

/**
 * Return beta as given by the definition of triangle barycentric coordinates
 * (beta is the coefficient for triangle vertex b in the equation
 * ndc_point = alpha * ndc_a + beta * ndc_b + gamma * ndc_c).
 */
float compute_beta(const s_coord &a, const s_coord &b, const s_coord &c,
                    const float x, const float y);

/**
 * Return gamma as given by the definition of triangle barycentric coordinates
 * (gamma is the coefficient for triangle vertex c in the equation
 * ndc_point = alpha * ndc_a + beta * ndc_b + gamma * ndc_c).
 */
float compute_gamma(const s_coord &a, const s_coord &b, const s_coord &c,
                    const float x, const float y);

/**
 * Raster a triangle face using the flat shading method (entire face is colored
 * with the color of the triangle barycentric average of triangle vertices and
 * corresponding vertex normals of vertices a, b, and c).
 */
void raster_flat_colored_triangle(const ndc_vertex &ndc_a,
                                     const ndc_vertex &ndc_b,
                                     const ndc_vertex &ndc_c,
                                     const color &color_avg,
                                     vector<vector<pixel>> &image,
                                     vector<vector<float>> &buffer_grid,
                                     const int xres, const int yres,
                                     const int max_intensity);

/**
 * Render a triangle face using the flat shading method (entire face is colored
 * with the color of the triangle barycentric average of triangle vertices and
 * corresponding vertex normals of vertices a, b, and c).
 */
void flat_shading(const vertex va, const vertex vb, const vertex vc,
                     const vnorm vna, const vnorm vnb, const vnorm vnc,
                     const object &obj, const vector<light> &lights,
                     const camera &cam, vector<vector<pixel>> &image,
                     vector<vector<float>> &buffer_grid, const int xres,
                     const int yres, const int max_intensity);

/**
 * Raster a triangle face using the Gouraud shading method (colors of each
 * point in the face are determined using interpolation via triangle barycentric
 * coordinates of the triangle vertices a, b, and c and their corresponding
 * colors).
 */
void raster_gouraud_colored_triangle(const ndc_vertex &ndc_a,
                                     const ndc_vertex &ndc_b,
                                     const ndc_vertex &ndc_c,
                                     const color &color_a,
                                     const color &color_b,
                                     const color &color_c,
                                     vector<vector<pixel>> &image,
                                     vector<vector<float>> &buffer_grid,
                                     const int xres, const int yres);

/**
 * Render a triangle face using the Gouraud shading method (colors of each
 * point in the face are determined using interpolation via triangle barycentric
 * coordinates of the triangle vertices a, b, and c and their corresponding
 * colors).
 */
void gouraud_shading(const vertex va, const vertex vb, const vertex vc,
                     const vnorm vna, const vnorm vnb, const vnorm vnc,
                     const object &obj, const vector<light> &lights,
                     const camera &cam, vector<vector<pixel>> &image,
                     vector<vector<float>> &buffer_grid, const int xres,
                     const int yres, const int max_intensity);

/**
 * Raster a triangle face using the Phong shading method (colors of each
 * point in the face are determined using interpolation via triangle barycentric
 * coordinates of the triangle vertices a, b, and c and their corresponding
 * vertex normals to determine the color of each point using the lighting model
 * for each point).
 */
void raster_phong_colored_triangle(const ndc_vertex &va, const vertex &vb,
                                   const vertex &vc, const vnorm &vna,
                                   const vnorm &vnb, const vnorm &vnc,
                                   const ndc_vertex &ndc_a,
                                   const ndc_vertex &ndc_b,
                                   const ndc_vertex &ndc_c,
                                   const camera &cam,
                                   const vector<light> &lights,
                                   const object &obj,
                                   vector<vector<pixel>> &image,
                                   vector<vector<float>> &buffer_grid,
                                   const int xres, const int yres,
                                   const int max_intensity);

/**
 * Render a triangle face using the Phong shading method (colors of each
 * point in the face are determined using interpolation via triangle barycentric
 * coordinates of the triangle vertices a, b, and c and their corresponding
 * vertex normals to determine the color of each point using the lighting model
 * for each point).
 */
void phong_shading(const vertex va, const vertex vb, const vertex vc,
                     const vnorm vna, const vnorm vnb, const vnorm vnc,
                     const object &obj, const vector<light> &lights,
                     const camera &cam, vector<vector<pixel>> &image,
                     vector<vector<float>> &buffer_grid, const int xres,
                     const int yres, const int max_intensity);
