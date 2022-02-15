// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// transformation.h

#include "general.h"

using namespace std;

/**
 * Return the transformation that transforms vectors from world space to camera
 * space. This transformation is represented by the matrix (TR)^-1, where R is
 * the rotation matrix that rotates vectors to the orientation specified by the
 * camera, and T is the translation matrix that translates vectors specified
 * by the camera's position.
 */
Matrix4f get_world_to_camera_transformation(position pos, orientation ori);

/**
 * Return the transformation that transforms vectors from camera space to
 * homogeneous normalized coordinate space (homogeneous NDC space). This
 * transformation is represented by the matrix (TR)^-1, where R is the
 * perspective_projection matrix that transforms vectors as specified by the
 * camera's perspective.
 */
Matrix4f get_perspective_projection_matrix(perspective per);

/**
 * Return an object transformed by its own given world space transformations.
 * All the object's vertices and vertex normals are transformed.
 */
object get_transformed_object(const object &obj);

/**
 * Return a vertex transformed from world space to Cartesian NDC space.
 */
ndc_vertex world_to_ndc(const vertex &w_vertex, const camera &cam);

/**
 * Return a vertex transformed from Cartesian NDC space to image screen space
 * (defined by a given xres by yres pixel grid).
 */
s_coord ndc_to_screen(const ndc_vertex &ndc_v, const int xres, const int yres);
