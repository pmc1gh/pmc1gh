// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// shaded_renderer.cpp

#include "scene_read.h"
#include "diagnostic.h"
#include "image_functions.h"
#include "shading.h"

using namespace std;
using namespace Eigen;

int main(int argc, char *argv []) {
    // Check to make sure there are exactly 4 command line arguments.
    if (argc != 5) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [scene_description_file.txt] [xres]"
                                 + " [yres] [mode]\n"
                                 + "mode == 0 => Gouraud shading\n"
                                 + "mode == 1 => Phong shading\n"
                                 + "Add \" | display -\" after"
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

    vector<vector<float>> buffer_grid = get_buffer_grid(xres, yres);

    int mode = atoi(argv[4]);

    camera cam = read_camera(argv[1]);

    vector<light> lights = read_lights(argv[1]);

    vector<object> objects = read_objects(argv[1]);

    // Transform all the objects with their given world space transformations.
    vector<object> t_objects;
    for (int i = 0; i < (int) objects.size(); i++) {
        t_objects.push_back(get_transformed_object(objects[i]));
    }

    for (int i = 0; i < (int) t_objects.size(); i++) {
        for (int j = 0; j < (int) t_objects[i].face_vector.size(); j++) {
            // Shade face j of object i.
            
            int v1_index = t_objects[i].face_vector[j].v1;
            int v2_index = t_objects[i].face_vector[j].v2;
            int v3_index = t_objects[i].face_vector[j].v3;
            int vn1_index = t_objects[i].face_vector[j].vn1;
            int vn2_index = t_objects[i].face_vector[j].vn2;
            int vn3_index = t_objects[i].face_vector[j].vn3;

            vertex va = t_objects[i].vertex_vector[v1_index];
            vertex vb = t_objects[i].vertex_vector[v2_index];
            vertex vc = t_objects[i].vertex_vector[v3_index];

            vnorm vna = t_objects[i].vnorm_vector[vn1_index];
            vnorm vnb = t_objects[i].vnorm_vector[vn2_index];
            vnorm vnc = t_objects[i].vnorm_vector[vn3_index];

            if (mode == 0) {
                gouraud_shading(va, vb, vc, vna, vnb, vnc, t_objects[i], lights,
                                cam, image, buffer_grid, xres, yres,
                                max_intensity);
            }
            if (mode == 1) {
                phong_shading(va, vb, vc, vna, vnb, vnc, t_objects[i], lights,
                                cam, image, buffer_grid, xres, yres,
                                max_intensity);
            }
        }
    }

    print_image(image, xres, yres, max_intensity);

    return 0;
}
