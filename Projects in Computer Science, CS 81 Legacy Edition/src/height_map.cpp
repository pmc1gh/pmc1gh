// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// height_map.cpp

#include <math.h>
#include <vector>
#include "scene_read.h"

using namespace std;

/////////////////////////////// Constructor ////////////////////////////////////

height_map::height_map() {
    min_x = -1;
    max_x = 1;
    min_y = -1;
    max_y = 1;
    x_res = 50; // 100
    y_res = 50; // 100

    // x and y space between vertices in the height_map
    float x_step = (max_x - min_x) / ((float) x_res - 1);
    float y_step = (max_y - min_y) / ((float) y_res - 1);

    hm_md.vertices = new vector<Vertex*>;
    hm_md.vertex_normals = new vector<Vec3f*>;
    hm_md.faces = new vector<Face*>;

    hm_md.vertices->push_back(NULL);
    hm_md.vertex_normals->push_back(NULL);

    for (int k = 0; k < 1; k++) {
        for (int j = 0; j < y_res; j++) {
            for (int i = 0; i < x_res; i++) {
                Vertex *v;
                v = new Vertex;

                /**
                 * Initialize the height_map as a flat sheet facing the
                 * +z-direction.
                 */
                Vec3f *n;
                n = new Vec3f;
                (*n).x = 0;
                (*n).y = 0;
                (*n).z = 1;

                (*v).x = min_x + i * x_step;
                (*v).y = min_y + j * y_step;
                (*v).z = 0;

                hm_md.vertices->push_back(v);
                hm_md.vertex_normals->push_back(n);

                if (i < x_res - 1 && j < y_res - 1) {
                    Face *f1, *f2;
                    /**
                     * In (row, col) coordinates, face f1 is the triangle
                     * oriented as (i, j) -> (i + 1, j) -> (i, j + 1), facing
                     * the +z-direction.
                     */
                    f1 = new Face;

                    /**
                     * In (row, col) coordinates, face f1 is the triangle
                     * oriented as (i + 1, j) -> (i + 1, j + 1) -> (i, j + 1),
                     * facing the +z-direction.
                     */
                    f2 = new Face;

                    if (k == 0) {
                        (*f1).vidx1 = j * y_res + i + 1;
                        (*f1).vidx2 = j * y_res + (i + 1) + 1;
                    }
                    else {
                        (*f1).vidx1 = j * y_res + (i + 1) + 1;
                        (*f1).vidx2 = j * y_res + i + 1;
                    }
                    (*f1).vidx3 = (j + 1) * y_res + i + 1;

                    (*f1).vnidx1 = (*f1).vidx1;
                    (*f1).vnidx2 = (*f1).vidx2;
                    (*f1).vnidx3 = (*f1).vidx3;

                    if (k == 0) {
                        (*f2).vidx1 = j * y_res + (i + 1) + 1;
                        (*f2).vidx2 = (j + 1) * y_res + (i + 1) + 1;
                    }
                    else {
                        (*f2).vidx1 = (j + 1) * y_res + (i + 1) + 1;
                        (*f2).vidx2 = j * y_res + (i + 1) + 1;
                    }
                    (*f2).vidx3 = (j + 1) * y_res + i + 1;

                    (*f2).vnidx1 = (*f2).vidx1;
                    (*f2).vnidx2 = (*f2).vidx2;
                    (*f2).vnidx3 = (*f2).vidx3;

                    hm_md.faces->push_back(f1);
                    hm_md.faces->push_back(f2);
                }
            }
        }
    }

    // 200x200 heighmap reaches max recursion depth in build_HE, I think...
    hevs = new vector<HEV*>();
    hefs = new vector<HEF*>();
    build_HE(&hm_md, hevs, hefs);

    // Update buffers below in same order after initialization here.
    for (int i = 0; i < (int) hm_md.faces->size(); i++) {
        Face *f = hm_md.faces->at(i);
        // Front side
        hm_obj.vertex_buffer.push_back(*(hm_md.vertices->at(f->vidx1)));
        hm_obj.vertex_buffer.push_back(*(hm_md.vertices->at(f->vidx2)));
        hm_obj.vertex_buffer.push_back(*(hm_md.vertices->at(f->vidx3)));

        Vec3f n_front1 = *(hm_md.vertex_normals->at(f->vnidx1));
        Vec3f n_front2 = *(hm_md.vertex_normals->at(f->vnidx2));
        Vec3f n_front3 = *(hm_md.vertex_normals->at(f->vnidx3));
        hm_obj.normal_buffer.push_back(n_front1);
        hm_obj.normal_buffer.push_back(n_front2);
        hm_obj.normal_buffer.push_back(n_front3);

        // Back side
        hm_obj.vertex_buffer.push_back(*(hm_md.vertices->at(f->vidx1)));
        hm_obj.vertex_buffer.push_back(*(hm_md.vertices->at(f->vidx3)));
        hm_obj.vertex_buffer.push_back(*(hm_md.vertices->at(f->vidx2)));

        Vec3f n_back1, n_back2, n_back3;
        n_back1.x = -n_front1.x;
        n_back1.y = -n_front1.y;
        n_back1.z = -n_front1.z;

        n_back2.x = -n_front2.x;
        n_back2.y = -n_front2.y;
        n_back2.z = -n_front2.z;

        n_back3.x = -n_front3.x;
        n_back3.y = -n_front3.y;
        n_back3.z = -n_front3.z;

        hm_obj.normal_buffer.push_back(n_back1);
        hm_obj.normal_buffer.push_back(n_back2);
        hm_obj.normal_buffer.push_back(n_back3);
    }

    float hm_color[3] = {0, 0, 1};

    hm_obj.ambient_reflect[0] = 0.1 * hm_color[0];
    hm_obj.ambient_reflect[1] = 0.1 * hm_color[1];
    hm_obj.ambient_reflect[2] = 0.1 * hm_color[2];
    hm_obj.diffuse_reflect[0] = hm_color[0];
    hm_obj.diffuse_reflect[1] = hm_color[1];
    hm_obj.diffuse_reflect[2] = hm_color[2];
    hm_obj.specular_reflect[0] = hm_color[0];
    hm_obj.specular_reflect[1] = hm_color[1];
    hm_obj.specular_reflect[2] = hm_color[2];
    hm_obj.shininess = 10;

    hm_obj.name = "Initial height_map";
}

////////////////////////// Accessor Methods ////////////////////////////////////

float height_map::get_min_x() const {
    return min_x;
}

float height_map::get_max_x() const {
    return max_x;
}

float height_map::get_min_y() const {
    return min_y;
}

float height_map::get_max_y() const {
    return max_y;
}

float height_map::get_x_res() const {
    return x_res;
}

float height_map::get_y_res() const {
    return y_res;
}

Mesh_Data height_map::get_hm_md() {
    return hm_md;
}

vector<HEV*> *height_map::get_hm_hevs() {
    return hevs;
}

vector<HEF*> *height_map::get_hm_hefs() {
    return hefs;
}

Object height_map::get_hm_obj() {
    return hm_obj;
}

////////////////////// height_map Update Function //////////////////////////////

/* Set the height_map's vertex z-values to the specified function and
 * recompute the normal vectors for hm_obj
 */
void height_map::set_hm(function<float(float, float)> hm_function) {
    for (int i = 1; i < (int) hm_md.vertices->size(); i++) {
        float x = hm_md.vertices->at(i)->x;
        float y = hm_md.vertices->at(i)->y;
        // Use the given height_map function to set the new z-value.
        hm_md.vertices->at(i)->z = hm_function(x, y);
    }

    delete_HE(hevs, hefs);
    hevs = new vector<HEV*>();
    hefs = new vector<HEF*>();
    build_HE(&hm_md, hevs, hefs);

    int buffer_index = 0;
    for (int i = 0; i < (int) hm_md.faces->size(); i++) {
        Face *f = hm_md.faces->at(i);
        // Front side
        hm_obj.vertex_buffer[buffer_index] = *(hm_md.vertices->at(f->vidx1));
        hm_obj.vertex_buffer[buffer_index+1] = *(hm_md.vertices->at(f->vidx2));
        hm_obj.vertex_buffer[buffer_index+2] = *(hm_md.vertices->at(f->vidx3));

        // Use the halfedge data structure to re-compute the vertex normals.
        Vec3f n_front1, n_front2, n_front3;
        n_front1 = calc_vertex_normal((*hevs)[f->vidx1]);
        n_front2 = calc_vertex_normal((*hevs)[f->vidx2]);
        n_front3 = calc_vertex_normal((*hevs)[f->vidx3]);
        hm_obj.normal_buffer[buffer_index] = n_front1;
        hm_obj.normal_buffer[buffer_index+1] = n_front2;
        hm_obj.normal_buffer[buffer_index+2] = n_front3;

        buffer_index += 3;

        // Back side
        hm_obj.vertex_buffer[buffer_index] = *(hm_md.vertices->at(f->vidx1));
        hm_obj.vertex_buffer[buffer_index+1] = *(hm_md.vertices->at(f->vidx3));
        hm_obj.vertex_buffer[buffer_index+2] = *(hm_md.vertices->at(f->vidx2));

        /**
         * Reverse the direction of the vertex normals computed above for the
         * back side vertices.
         */
        Vec3f n_back1, n_back2, n_back3;
        n_back1.x = -n_front1.x;
        n_back1.y = -n_front1.y;
        n_back1.z = -n_front1.z;

        n_back2.x = -n_front2.x;
        n_back2.y = -n_front2.y;
        n_back2.z = -n_front2.z;

        n_back3.x = -n_front3.x;
        n_back3.y = -n_front3.y;
        n_back3.z = -n_front3.z;

        hm_obj.normal_buffer[buffer_index] = n_back1;
        hm_obj.normal_buffer[buffer_index+1] = n_back2;
        hm_obj.normal_buffer[buffer_index+2] = n_back3;

        buffer_index += 3;
    }
    hm_obj.name = "Modified height_map";
}
