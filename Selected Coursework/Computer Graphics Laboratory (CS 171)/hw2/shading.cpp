// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// shading.cpp

#include "shading.h"

using namespace std;

/**
 * Return the color of a given vertex with a given vertex normal vector
 * determined by the implemented lighting model.
 */
color lighting(const vertex v, const vnorm vn, const object &obj,
               const vector<light> &lights, const camera &cam) {
    Vector3f vn_v;
    vn_v << vn.x, vn.y, vn.z;
    vn_v = vn_v.normalized();

    color cd = obj.diff;
    color ca = obj.ambi;
    color cs = obj.spec;
    float s = obj.shin;

    Vector3f diffuse_sum;
    Vector3f specular_sum;

    diffuse_sum << 0, 0, 0;
    specular_sum << 0, 0, 0;

    Vector3f cam_v_vec;
    cam_v_vec << cam.pos.x - v.x, cam.pos.y - v.y, cam.pos.z - v.z;
    Vector3f cam_dir = cam_v_vec.normalized();

    Vector3f pos;
    pos << v.x, v.y, v.z;

    for (int i = 0; i < (int) lights.size(); i++) {
        light l = lights[i];
        Vector3f lp;
        Vector3f lc;
        Vector3f l_dir;

        lp << l.x, l.y, l.z;
        lc << l.r, l.g, l.b;
        l_dir << lp - pos;
        float dSquared = l_dir.squaredNorm();
        lc /= (1 + l.k * dSquared);
        l_dir = l_dir.normalized();

        Vector3f l_diff = lc * max((float) 0.0, (float) vn_v.dot(l_dir));
        diffuse_sum += l_diff;

        Vector3f dir_sum = cam_dir + l_dir;
        Vector3f l_spec = lc * pow(max((float) 0.0, (float)
                                            vn_v.dot(dir_sum.normalized())), s);
        specular_sum += l_spec;
    }

    color c;
    c.r = min((float) 1.0,
              ca.r + diffuse_sum(0,0) * cd.r + specular_sum(0,0) * cs.r);
    c.g = min((float) 1.0,
              ca.g + diffuse_sum(1,0) * cd.g + specular_sum(1,0) * cs.g);
    c.b = min((float) 1.0,
              ca.b + diffuse_sum(2,0) * cd.b + specular_sum(2,0) * cs.b);
    return c;
}

/**
 * Helper function for computing alpha, beta, and gamma used in triangle
 * barycentric coordinates.
 */
float compute_f_ij(const float xi, const float yi, const float xj,
                   const float yj, const float x, const float y) {
    return (yi - yj) * x + (xj - xi) * y + xi * yj - xj * yi;
}

/**
 * Return alpha as given by the definition of triangle barycentric coordinates
 * (alpha is the coefficient for triangle vertex a in the equation
 * ndc_point = alpha * ndc_a + beta * ndc_b + gamma * ndc_c).
 */
float compute_alpha(const s_coord &a, const s_coord &b, const s_coord &c,
                    const float x, const float y) {
    return compute_f_ij(b.x, b.y, c.x, c.y, x, y)
           / compute_f_ij(b.x, b.y, c.x, c.y, a.x, a.y);
}

/**
 * Return beta as given by the definition of triangle barycentric coordinates
 * (beta is the coefficient for triangle vertex b in the equation
 * ndc_point = alpha * ndc_a + beta * ndc_b + gamma * ndc_c).
 */
float compute_beta(const s_coord &a, const s_coord &b, const s_coord &c,
                    const float x, const float y) {
    return compute_f_ij(a.x, a.y, c.x, c.y, x, y)
           / compute_f_ij(a.x, a.y, c.x, c.y, b.x, b.y);
}

/**
 * Return gamma as given by the definition of triangle barycentric coordinates
 * (gamma is the coefficient for triangle vertex c in the equation
 * ndc_point = alpha * ndc_a + beta * ndc_b + gamma * ndc_c).
 */
float compute_gamma(const s_coord &a, const s_coord &b, const s_coord &c,
                    const float x, const float y) {
    return compute_f_ij(a.x, a.y, b.x, b.y, x, y)
           / compute_f_ij(a.x, a.y, b.x, b.y, c.x, c.y);
}

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
                                     const int max_intensity) {
    // Backface culling.
    Vector3f b_to_c;
    Vector3f b_to_a;

    b_to_c << ndc_c.x - ndc_b.x, ndc_c.y - ndc_b.y,
              ndc_c.z - ndc_b.z;
    b_to_a << ndc_a.x - ndc_b.x, ndc_a.y - ndc_b.y,
              ndc_a.z - ndc_b.z;

    Vector3f cross = b_to_c.cross(b_to_a);

    if (cross(2,0) < 0) {
        return;
    }

    s_coord a = ndc_to_screen(ndc_a, xres, yres);
    s_coord b = ndc_to_screen(ndc_b, xres, yres);
    s_coord c = ndc_to_screen(ndc_c, xres, yres);

    int x_min = min(min(a.x, b.x), c.x);
    int x_max = max(max(a.x, b.x), c.x);
    int y_min = min(min(a.y, b.y), c.y);
    int y_max = max(max(a.y, b.y), c.y);

    for (int x = x_min; x <= x_max; x++) {
        for (int y = y_min; y <= y_max; y++) {
            /* alpha, beta, and gamma for triangle interpolation via barycentric
             * coordinates.
             */
            float alpha = compute_alpha(a, b, c, x, y);
            float beta = compute_beta(a, b, c, x, y);
            float gamma = compute_gamma(a, b, c, x, y);

            if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1
                && gamma >= 0 && gamma <= 1){
                // Depth buffering.
                ndc_vertex ndc;
                ndc.x = alpha * ndc_a.x + beta * ndc_b.x + gamma * ndc_c.x;
                ndc.y = alpha * ndc_a.y + beta * ndc_b.y + gamma * ndc_c.y;
                ndc.z = alpha * ndc_a.z + beta * ndc_b.z + gamma * ndc_c.z;
                if (ndc.x > -1 && ndc.x < 1 && ndc.y > -1 && ndc.y < 1
                    && ndc.z > -1 && ndc.z < 1) {
                    if (ndc.z < buffer_grid[y][x]) {
                        buffer_grid[y][x] = ndc.z;
                        float red = (float) max_intensity * color_avg.r;
                        float green = (float) max_intensity * color_avg.g;
                        float blue = (float) max_intensity * color_avg.b;

                        image[y][x].r = (int) red;
                        image[y][x].g = (int) green;
                        image[y][x].b = (int) blue;
                    }
                }
            }
        }
    }
}

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
                     const int yres, const int max_intensity) {
    vertex v_avg;
    v_avg.x = (float) (va.x + vb.x + vc.x) / 3;
    v_avg.y = (float) (va.y + vb.y + vc.y) / 3;
    v_avg.z = (float) (va.z + vb.z + vc.z) / 3;

    vnorm vn_avg;
    vn_avg.x = (float) (vna.x + vnb.x + vnc.x) / 3;
    vn_avg.y = (float) (vna.y + vnb.y + vnc.y) / 3;
    vn_avg.z = (float) (vna.z + vnb.z + vnc.z) / 3;
    color color_avg = lighting(v_avg, vn_avg, obj, lights, cam);

    ndc_vertex ndc_a = world_to_ndc(va, cam);
    ndc_vertex ndc_b = world_to_ndc(vb, cam);
    ndc_vertex ndc_c = world_to_ndc(vc, cam);

    raster_flat_colored_triangle(ndc_a, ndc_b, ndc_c, color_avg, image,
                                 buffer_grid, xres, yres, max_intensity);
}

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
                                     const int xres, const int yres,
                                     const int max_intensity) {
    // Backface culling.
    Vector3f b_to_c;
    Vector3f b_to_a;

    b_to_c << ndc_c.x - ndc_b.x, ndc_c.y - ndc_b.y,
              ndc_c.z - ndc_b.z;
    b_to_a << ndc_a.x - ndc_b.x, ndc_a.y - ndc_b.y,
              ndc_a.z - ndc_b.z;

    Vector3f cross = b_to_c.cross(b_to_a);

    if (cross(2,0) < 0) {
        return;
    }

    s_coord a = ndc_to_screen(ndc_a, xres, yres);
    s_coord b = ndc_to_screen(ndc_b, xres, yres);
    s_coord c = ndc_to_screen(ndc_c, xres, yres);

    int x_min = min(min(a.x, b.x), c.x);
    int x_max = max(max(a.x, b.x), c.x);
    int y_min = min(min(a.y, b.y), c.y);
    int y_max = max(max(a.y, b.y), c.y);

    for (int x = x_min; x <= x_max; x++) {
        for (int y = y_min; y <= y_max; y++) {
            /* alpha, beta, and gamma for triangle interpolation via barycentric
             * coordinates.
             */
            float alpha = compute_alpha(a, b, c, x, y);
            float beta = compute_beta(a, b, c, x, y);
            float gamma = compute_gamma(a, b, c, x, y);

            if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1
                && gamma >= 0 && gamma <= 1){
                // Depth buffering.
                ndc_vertex ndc;
                ndc.x = alpha * ndc_a.x + beta * ndc_b.x + gamma * ndc_c.x;
                ndc.y = alpha * ndc_a.y + beta * ndc_b.y + gamma * ndc_c.y;
                ndc.z = alpha * ndc_a.z + beta * ndc_b.z + gamma * ndc_c.z;
                if (ndc.x > -1 && ndc.x < 1 && ndc.y > -1 && ndc.y < 1
                    && ndc.z > -1 && ndc.z < 1) {
                    if (ndc.z < buffer_grid[y][x]) {
                        buffer_grid[y][x] = ndc.z;
                        float red = (float) max_intensity * (alpha * color_a.r
                            + beta * color_b.r + gamma * color_c.r);
                        float green = (float) max_intensity * (alpha * color_a.g
                            + beta * color_b.g + gamma * color_c.g);
                        float blue = (float) max_intensity * (alpha * color_a.b
                            + beta * color_b.b + gamma * color_c.b);

                        image[y][x].r = (int) red;
                        image[y][x].g = (int) green;
                        image[y][x].b = (int) blue;
                    }
                 }
             }
         }
     }
}

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
                     const int yres, const int max_intensity) {
    color color_a = lighting(va, vna, obj, lights, cam);
    color color_b = lighting(vb, vnb, obj, lights, cam);
    color color_c = lighting(vc, vnc, obj, lights, cam);

    ndc_vertex ndc_a = world_to_ndc(va, cam);
    ndc_vertex ndc_b = world_to_ndc(vb, cam);
    ndc_vertex ndc_c = world_to_ndc(vc, cam);

    raster_gouraud_colored_triangle(ndc_a, ndc_b, ndc_c, color_a, color_b,
                                    color_c, image, buffer_grid, xres, yres,
                                    max_intensity);
}

/**
 * Raster a triangle face using the Phong shading method (colors of each
 * point in the face are determined using interpolation via triangle barycentric
 * coordinates of the triangle vertices a, b, and c and their corresponding
 * vertex normals to determine the color of each point using the lighting model
 * for each point).
 */
void raster_phong_colored_triangle(const vertex &va, const vertex &vb,
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
                                   const int max_intensity) {
    // Backface culling.
    Vector3f b_to_c;
    Vector3f b_to_a;

    b_to_c << ndc_c.x - ndc_b.x, ndc_c.y - ndc_b.y,
              ndc_c.z - ndc_b.z;
    b_to_a << ndc_a.x - ndc_b.x, ndc_a.y - ndc_b.y,
              ndc_a.z - ndc_b.z;

    Vector3f cross = b_to_c.cross(b_to_a);

    if (cross(2,0) < 0) {
        return;
    }

    s_coord a = ndc_to_screen(ndc_a, xres, yres);
    s_coord b = ndc_to_screen(ndc_b, xres, yres);
    s_coord c = ndc_to_screen(ndc_c, xres, yres);

    int x_min = min(min(a.x, b.x), c.x);
    int x_max = max(max(a.x, b.x), c.x);
    int y_min = min(min(a.y, b.y), c.y);
    int y_max = max(max(a.y, b.y), c.y);

    for (int x = x_min; x <= x_max; x++) {
        for (int y = y_min; y <= y_max; y++) {
            // alpha, beta, and gamma for triangle barycentric coordinates.
            float alpha = compute_alpha(a, b, c, x, y);
            float beta = compute_beta(a, b, c, x, y);
            float gamma = compute_gamma(a, b, c, x, y);

            if (alpha >= 0 && alpha <= 1 && beta >= 0 && beta <= 1
                && gamma >= 0 && gamma <= 1){

                // Depth buffering.
                ndc_vertex ndc;
                ndc.x = alpha * ndc_a.x + beta * ndc_b.x + gamma * ndc_c.x;
                ndc.y = alpha * ndc_a.y + beta * ndc_b.y + gamma * ndc_c.y;
                ndc.z = alpha * ndc_a.z + beta * ndc_b.z + gamma * ndc_c.z;
                if (ndc.x > -1 && ndc.x < 1 && ndc.y > -1 && ndc.y < 1
                    && ndc.z > -1 && ndc.z < 1) {
                    if (ndc.z < buffer_grid[y][x]) {
                        buffer_grid[y][x] = ndc.z;
                        vertex v;
                        vnorm vn;

                        v.x = alpha * va.x + beta * vb.x + gamma * vc.x;
                        v.y = alpha * va.y + beta * vb.y + gamma * vc.y;
                        v.z = alpha * va.z + beta * vb.z + gamma * vc.z;

                        vn.x = alpha * vna.x + beta * vnb.x + gamma * vnc.x;
                        vn.y = alpha * vna.y + beta * vnb.y + gamma * vnc.y;
                        vn.z = alpha * vna.z + beta * vnb.z + gamma * vnc.z;

                        color c = lighting(v, vn, obj, lights, cam);

                        image[y][x].r = (int) ((float) max_intensity * c.r);
                        image[y][x].g = (int) ((float) max_intensity * c.g);
                        image[y][x].b = (int) ((float) max_intensity * c.b);
                    }
                 }
             }
         }
     }
}

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
                     const int yres, const int max_intensity) {
    ndc_vertex ndc_a = world_to_ndc(va, cam);
    ndc_vertex ndc_b = world_to_ndc(vb, cam);
    ndc_vertex ndc_c = world_to_ndc(vc, cam);

    raster_phong_colored_triangle(va, vb, vc, vna, vnb, vnc, ndc_a, ndc_b,
                                  ndc_c, cam, lights, obj, image, buffer_grid,
                                  xres, yres, max_intensity);
}
