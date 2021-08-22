// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// default_scene.cpp

#include <vector>
#include "scene_read.h"

using namespace std;

/**
 * Initialze a light for the lights vector for the scene.
 */
void create_default_lights(vector<Point_Light> &lights) {
    Point_Light light0;
    light0.pos[0] = -0.8;
    light0.pos[1] = 0.0;
    light0.pos[2] = 2.0;
    light0.pos[3] = 1.0;
    light0.color[0] = 1.0;
    light0.color[1] = 1.0;
    light0.color[2] = 1.0;
    light0.k = 0.2;
    lights.push_back(light0);
}

/**
 * Initialize the camera for the scene.
 */
void create_default_camera(Camera &cam) {
    cam.pos[0] = 0.0;
    cam.pos[1] = 0.0;
    cam.pos[2] = 3.0;
    cam.ori_axis[0] = 0.0;
    cam.ori_axis[1] = 0.0;
    cam.ori_axis[2] = 1.0;
    cam.ori_angle = 0.0;
    cam.n = 1;
    cam.f = 20;
    cam.l = -0.5;
    cam.r = 0.5;
    cam.t = 0.5;
    cam.b = -0.5;
}

/**
 * Create a square to place on the texture image.
 */
void create_default_square(vector<Object> &objects) {
    Object square;

    square.texture_image = "../data/textures/lion.png";
    square.texture_normals = "../data/textures/lion-normals.png";

    ////////////////////////////////////////////////////////////////////////////
    // Reflectances
    ////////////////////////////////////////////////////////////////////////////

    square.ambient_reflect[0] = 0.2;
    square.ambient_reflect[1] = 0.2;
    square.ambient_reflect[2] = 0.2;

    square.diffuse_reflect[0] = 0.6;
    square.diffuse_reflect[1] = 0.6;
    square.diffuse_reflect[2] = 0.6;

    square.specular_reflect[0] = 1;
    square.specular_reflect[1] = 1;
    square.specular_reflect[2] = 1;

    square.shininess = 5.0;

    ////////////////////////////////////////////////////////////////////////////
    // Texture Coordinates and Tangent Vectors
    ////////////////////////////////////////////////////////////////////////////

    vector<TextureCoords> texture_buffer;
    TextureCoords tc;
    tc.u = 0;
    tc.v = 0;
    texture_buffer.push_back(tc);
    tc.u = 1;
    tc.v = 0;
    texture_buffer.push_back(tc);
    tc.u = 1;
    tc.v = 1;
    texture_buffer.push_back(tc);
    tc.u = 0;
    tc.v = 0;
    texture_buffer.push_back(tc);
    tc.u = 1;
    tc.v = 1;
    texture_buffer.push_back(tc);
    tc.u = 0;
    tc.v = 1;
    texture_buffer.push_back(tc);
    // other side
    tc.u = 0;
    tc.v = 0;
    texture_buffer.push_back(tc);
    tc.u = 1;
    tc.v = 1;
    texture_buffer.push_back(tc);
    tc.u = 1;
    tc.v = 0;
    texture_buffer.push_back(tc);
    tc.u = 0;
    tc.v = 0;
    texture_buffer.push_back(tc);
    tc.u = 0;
    tc.v = 1;
    texture_buffer.push_back(tc);
    tc.u = 1;
    tc.v = 1;
    texture_buffer.push_back(tc);

    vector<Vec3f> tangent_buffer;
    Vec3f tangent;
    tangent.x = -1;
    tangent.y = 0;
    tangent.z = 0;
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    // other side
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);
    tangent_buffer.push_back(tangent);

    square.texture_coords_buffer = texture_buffer;
    square.tangent_buffer = tangent_buffer;

    ////////////////////////////////////////////////////////////////////////////
    // Points
    ////////////////////////////////////////////////////////////////////////////

    Vertex point1;
    point1.x = -1;
    point1.y = -1;
    point1.z = 0;

    Vertex point2;
    point2.x = 1;
    point2.y = -1;
    point2.z = 0;

    Vertex point3;
    point3.x = 1;
    point3.y = 1;
    point3.z = 0;

    Vertex point4;
    point4.x = -1;
    point4.y = 1;
    point4.z = 0;

    ////////////////////////////////////////////////////////////////////////////
    // Normals
    ////////////////////////////////////////////////////////////////////////////

    Vec3f normal1;
    normal1.x = 0;
    normal1.y = 0;
    normal1.z = 1;

    ////////////////////////////////////////////////////////////////////////////
    // Vertex and Normal Arrays
    ////////////////////////////////////////////////////////////////////////////

    /* Face 1: */

    square.vertex_buffer.push_back(point1);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point2);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point3);
    square.normal_buffer.push_back(normal1);

    /* Face 2: */

    square.vertex_buffer.push_back(point1);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point3);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point4);
    square.normal_buffer.push_back(normal1);

    /* Face 3: */

    square.vertex_buffer.push_back(point1);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point3);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point2);
    square.normal_buffer.push_back(normal1);

    /* Face 4: */

    square.vertex_buffer.push_back(point1);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point4);
    square.normal_buffer.push_back(normal1);

    square.vertex_buffer.push_back(point3);
    square.normal_buffer.push_back(normal1);
    objects.push_back(square);
}
