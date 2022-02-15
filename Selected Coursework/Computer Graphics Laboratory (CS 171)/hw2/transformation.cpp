// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// transformation.cpp

#include "transformation.h"

using namespace std;

/**
 * Return the transformation that transforms vectors from world space to camera
 * space. This transformation is represented by the matrix (TR)^-1, where R is
 * the rotation matrix that rotates vectors to the orientation specified by the
 * camera, and T is the translation matrix that translates vectors specified
 * by the camera's position.
 */
Matrix4f get_world_to_camera_transformation(position pos, orientation ori) {
    float rx = ori.rx;
    float ry = ori.ry;
    float rz = ori.rz;
    float angle = ori.angle;
    float a00 = rx * rx + (1 - rx * rx) * cos(angle);
    float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
    float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
    float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
    float a11 = ry * ry + (1 - ry * ry) * cos(angle);
    float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
    float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
    float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
    float a22 = rz * rz + (1 - rz * rz) * cos(angle);
    Matrix4f rotation_matrix;
    rotation_matrix << a00, a01, a02, 0,
                       a10, a11, a12, 0,
                       a20, a21, a22, 0,
                       0, 0, 0, 1;

    Matrix4f translation_matrix;
    translation_matrix << 1, 0, 0, pos.x,
                          0, 1, 0, pos.y,
                          0, 0, 1, pos.z,
                          0, 0, 0, 1;

    Matrix4f product_matrix = translation_matrix * rotation_matrix;
    Matrix4f w_to_c_transformation = product_matrix.inverse();
    return w_to_c_transformation;
}

/**
 * Return the transformation that transforms vectors from camera space to
 * homogeneous normalized coordinate space (homogeneous NDC space). This
 * transformation is represented by the matrix (TR)^-1, where R is the
 * perspective_projection matrix that transforms vectors as specified by the
 * camera's perspective.
 */
Matrix4f get_perspective_projection_matrix(perspective per) {
    Matrix4f perspective_projection;
    float a00 = 2 * per.n / (per.r - per.l);
    float a02 = (per.r + per.l) / (per.r - per.l);
    float a11 = 2 * per.n / (per.t - per.b);
    float a12 = (per.t + per.b) / (per.t - per.b);
    float a22 = -(per.f + per.n) / (per.f - per.n);
    float a23 = -2 * per.f * per.n / (per.f - per.n);
    perspective_projection << a00, 0, a02, 0,
                              0, a11, a12, 0,
                              0, 0, a22, a23,
                              0, 0, -1, 0;
    return perspective_projection;
}

/**
 * Return an object transformed by its own given world space transformations.
 * All the object's vertices and vertex normals are transformed.
 */
object get_transformed_object(const object &obj) {
    Matrix4f v_transform;
    v_transform << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

    Matrix4f n_transform;
    n_transform << 1, 0, 0, 0,
                   0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

    // Multiply all the transformations together into one transformation.
    vector<Matrix4f> tm_vector = obj.t_transform_vector;
    for (int i = 0; i < (int) tm_vector.size(); i++) {
        v_transform = tm_vector[i] * v_transform;
    }

    vector<Matrix4f> nm_vector = obj.n_transform_vector;
    for (int i = 0; i < (int) nm_vector.size(); i++) {
        n_transform = nm_vector[i] * n_transform;
    }

    Matrix4f n_transform_inv = n_transform.inverse();
    Matrix4f vn_transform = n_transform_inv.transpose();

    object transformed_object = obj;

    vector<vertex> tv_vector;
    vector<vnorm> tvn_vector;

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    vertex null_vertex;
    null_vertex.x = 0;
    null_vertex.y = 0;
    null_vertex.z = 0;
    tv_vector.push_back(null_vertex);

    /* Append a "null" vnorm as the 0th element of the vnorm vector so other
     * vnorms can be 1-indexed.
     */
    vnorm null_vnorm;
    null_vnorm.x = 0;
    null_vnorm.y = 0;
    null_vnorm.z = 0;
    tvn_vector.push_back(null_vnorm);

    // Transform all the vertices in the given vector of vertices.
    vector<vertex> v_vector = obj.vertex_vector;

    for (int i = 1; i < (int) v_vector.size(); i++) {
        Vector4f v;
        v << v_vector[i].x, v_vector[i].y, v_vector[i].z, 1.0;
        Vector4f tv;
        tv = v_transform * v;
        vertex t_vertex;
        t_vertex.x = tv(0) / tv(3);
        t_vertex.y = tv(1) / tv(3);
        t_vertex.z = tv(2) / tv(3);
        tv_vector.push_back(t_vertex);
    }

    vector<vnorm> vn_vector = obj.vnorm_vector;
    for (int i = 1; i < (int) vn_vector.size(); i++) {
        Vector4f vn;
        vn << vn_vector[i].x, vn_vector[i].y, vn_vector[i].z, 1.0;
        Vector4f tvn;
        tvn = vn_transform * vn;
        vnorm t_vnorm;
        t_vnorm.x = tvn(0) / tvn(3);
        t_vnorm.y = tvn(1) / tvn(3);
        t_vnorm.z = tvn(2) / tvn(3);
        tvn_vector.push_back(t_vnorm);
    }

    transformed_object.vertex_vector = tv_vector;
    transformed_object.vnorm_vector = tvn_vector;
    return transformed_object;
}

/**
 * Return a vertex transformed from world space to Cartesian NDC space.
 */
ndc_vertex world_to_ndc(const vertex &w_vertex, const camera &cam) {
    Matrix4f camera_space_transform =
        get_world_to_camera_transformation(cam.pos, cam.ori);

    Matrix4f perspective_transform =
        get_perspective_projection_matrix(cam.per);

    Matrix4f transform;
    transform = perspective_transform * camera_space_transform;

    Vector4f w_vector;
    w_vector << w_vertex.x, w_vertex.y, w_vertex.z, 1.0;

    Vector4f ndc_vector = transform * w_vector;
    ndc_vector /= ndc_vector(3,0);

    ndc_vertex ndc_v;
    ndc_v.x = ndc_vector(0,0);
    ndc_v.y = ndc_vector(1,0);
    ndc_v.z = ndc_vector(2,0);

    return ndc_v;
}

/**
 * Return a vertex transformed from Cartesian NDC space to image screen space
 * (defined by a given xres by yres pixel grid).
 */
s_coord ndc_to_screen(const ndc_vertex &ndc_v, const int xres, const int yres) {
    float x = ndc_v.x;
    float y = ndc_v.y;
    s_coord sc;
    sc.x = (int) ((x + 1) * ((xres - 1) / 2));
    sc.y = (int) ((-y + 1) * ((yres - 1) / 2));
    return sc;
}
