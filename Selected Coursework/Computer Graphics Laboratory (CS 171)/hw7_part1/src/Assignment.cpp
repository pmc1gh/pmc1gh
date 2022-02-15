// Philip Carr
// CS 171 Assignment 7 Part 1
// December 13, 2018
// Assignment.cpp

#include "Assignment.hpp"

#include "model.hpp"
#include "UI.hpp"
#include "Scene.hpp"
#include "PNGMaker.hpp"

#define XRES 300 // 250 originally
#define YRES 300 // 250 originally

using namespace std;

namespace Assignment {
    /* Assignment Part A, Part 1 */
    static const float k_min_io_test_x = -7.0;
    static const float k_max_io_test_x = 7.0;
    static const float k_min_io_test_y = -7.0;
    static const float k_max_io_test_y = 7.0;
    static const float k_min_io_test_z = -7.0;
    static const float k_max_io_test_z = 7.0;
    static const int k_io_test_x_count = 15;
    static const int k_io_test_y_count = 15;
    static const int k_io_test_z_count = 15;

    /**
     * General-purpose function for obtaining the total transformation matrix
     * from a transformation stack. The additional boolean arguments change
     * whether the transformation is the regular, inverse, or inverse transpose
     * transformation, and whether translations, scaling, and/or rotations are
     * included in the total transformation returned.
     */
    Matrix4f getTotalTransform(vector<Transformation> &transformation_stack,
                               bool get_inverse, bool get_inverse_transpose,
                               bool with_translations, bool with_scaling,
                               bool with_rotations) {
        Matrix4f identity_transform;
        identity_transform << 1, 0, 0, 0,
                              0, 1, 0, 0,
                              0, 0, 1, 0,
                              0, 0, 0, 1;
        /* Create the total transformation of the transformations contained in
         * transformation_stack.
         */
        Matrix4f transform = identity_transform;
        for (int i = 0; i < (int) transformation_stack.size(); i++) {
            Transformation next_transform = transformation_stack[i];
            Matrix4f next_transform_matrix;
            if (next_transform.type == TRANS) {
                if (with_translations) {
                    next_transform_matrix <<
                        1, 0, 0, next_transform.trans(0),
                        0, 1, 0, next_transform.trans(1),
                        0, 0, 1, next_transform.trans(2),
                        0, 0, 0, 1;
                }
                else {
                    next_transform_matrix = identity_transform;
                }
            }
            else if (next_transform.type == SCALE) {
                if (with_scaling) {
                    next_transform_matrix <<
                        next_transform.trans(0), 0, 0, 0,
                        0, next_transform.trans(1), 0, 0,
                        0, 0, next_transform.trans(2), 0,
                        0, 0, 0, 1;
                }
                else {
                    next_transform_matrix = identity_transform;
                }
            }
            else {
                if (with_rotations) {
                    float rx = next_transform.trans(0);
                    float ry = next_transform.trans(1);
                    float rz = next_transform.trans(2);
                    float angle = next_transform.trans(3);

                    float rotation_axis_magnitude =
                        sqrt(rx * rx + ry * ry + rz * rz);
                    rx /= rotation_axis_magnitude;
                    ry /= rotation_axis_magnitude;
                    rz /= rotation_axis_magnitude;

                    float a00 = rx * rx + (1 - rx * rx) * cos(angle);
                    float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
                    float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
                    float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
                    float a11 = ry * ry + (1 - ry * ry) * cos(angle);
                    float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
                    float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
                    float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
                    float a22 = rz * rz + (1 - rz * rz) * cos(angle);

                    next_transform_matrix <<
                        a00, a01, a02, 0,
                        a10, a11, a12, 0,
                        a20, a21, a22, 0,
                        0, 0, 0, 1;
                }
                else {
                    next_transform_matrix = identity_transform;
                }
            }
            transform *= next_transform_matrix;
        }

        if (get_inverse) {
            Matrix4f inverse_transform = transform.inverse();
            return inverse_transform;
        }
        if (get_inverse_transpose) {
            Matrix4f inverse_transform = transform.inverse();
            Matrix4f inverse_transpose_transform =
                inverse_transform.transpose();
            return inverse_transpose_transform;
        }

        return transform;
    }

    bool IOTest(
        Renderable *ren,
        vector<Transformation> &transformation_stack,
        float x,
        float y,
        float z)
    {
        if (ren->getType() == PRM) {
            Primitive *prm = dynamic_cast<Primitive*>(ren);

            Matrix4f inverse_transform =
                getTotalTransform(transformation_stack, true, false, true, true,
                                  true);

            // Apply the inverse transformation to the given (x, y, z) point.
            Vector4f t_vec;
            t_vec << x, y, z, 1;

            /* Inverse-transform the current point to the origin coordinates of
             * the superquadric.
             */
            t_vec = inverse_transform * t_vec;

            float t_x = t_vec(0) / prm->getCoeff()(0);
            float t_y = t_vec(1) / prm->getCoeff()(1);
            float t_z = t_vec(2) / prm->getCoeff()(2);

            float e_val = prm->getExp0();
            float n_val = prm->getExp1();

            float superquadric_val =
                pow(pow(pow(t_x, 2.0), 1.0 / e_val)
                    + pow(pow(t_y, 2.0), 1.0 / e_val), e_val / n_val)
                + pow(pow(t_z, 2.0), 1.0 / n_val) - 1.0;

            /* Return true if the point is inside the primitive. Otherwise
             * return false.
             */
            return superquadric_val < 0;

        } else if (ren->getType() == OBJ) {
            Object *obj = dynamic_cast<Object*>(ren);
            const vector<Transformation>& overall_trans =
                obj->getOverallTransformation();
            for (int i = overall_trans.size() - 1; i >= 0; i--) {
                transformation_stack.push_back(overall_trans.at(i));
            }

            bool IO_result = false;
            for (auto& child_it : obj->getChildren()) {
                const vector<Transformation>& child_trans =
                    child_it.second.transformations;
                for (int i = child_trans.size() - 1; i >= 0; i--) {
                    transformation_stack.push_back(child_trans.at(i));
                }
                IO_result |= IOTest(
                    Renderable::get(child_it.second.name),
                    transformation_stack,
                    x, y, z);
                transformation_stack.erase(
                    transformation_stack.end() - child_trans.size(),
                    transformation_stack.end());
            }

            transformation_stack.erase(
                transformation_stack.end() - overall_trans.size(),
                transformation_stack.end());
            return IO_result;
        } else {
            fprintf(stderr, "Renderer::draw ERROR invalid RenderableType %d\n",
                ren->getType());
            exit(1);
        }

        return true;
    }

    void drawIOTest() {
        const Line* cur_state = CommandLine::getState();
        Renderable* current_selected_ren = NULL;

        if (cur_state) {
            current_selected_ren = Renderable::get(cur_state->tokens[1]);
        }

        if (current_selected_ren == NULL) {
            return;
        }

        const float IO_test_color[3] = {0.5, 0.0, 0.0};
        glMaterialfv(GL_FRONT, GL_AMBIENT, IO_test_color);
        for (int x = 0; x < k_io_test_x_count; x++) {
            for (int y = 0; y < k_io_test_y_count; y++) {
                for (int z = 0; z < k_io_test_z_count; z++) {
                    float test_x = k_min_io_test_x
                        + x * (k_max_io_test_x - k_min_io_test_x)
                            / (float) k_io_test_x_count;
                    float test_y = k_min_io_test_y
                        + y * (k_max_io_test_y - k_min_io_test_y)
                            / (float) k_io_test_y_count;
                    float test_z = k_min_io_test_z
                        + z * (k_max_io_test_z - k_min_io_test_z)
                            / (float) k_io_test_z_count;

                    vector<Transformation> transformation_stack;
                    if (IOTest(
                            current_selected_ren,
                            transformation_stack,
                            test_x,
                            test_y,
                            test_z))
                    {
                        glPushMatrix();
                        glTranslatef(test_x, test_y, test_z);
                        glutWireSphere(0.05, 4, 4);
                        glPopMatrix();
                    }
                }
            }
        }
    }

    /* Assignment Part A, Part 2 */
    struct Ray {
        // Boolean to tell if ray is actually represents an intersection.
        bool is_intersect;

        float origin_x;
        float origin_y;
        float origin_z;

        float direction_x;
        float direction_y;
        float direction_z;

        Vector3f getLocation(float t) {
            Vector3f loc;
            loc << origin_x + t * direction_x,
                origin_y + t * direction_y,
                origin_z + t * direction_z;
            return loc;
        }
    };

    Ray findIntersection(Primitive *prm,
                         vector<Transformation> transformation_stack,
                         const Ray &camera_ray) {
        // Initialize the intersection_ray.
        Ray intersection_ray;
        intersection_ray.is_intersect = false;
        intersection_ray.origin_x = 0.0;
        intersection_ray.origin_y = 0.0;
        intersection_ray.origin_z = 0.0;
        intersection_ray.direction_x = 0.0;
        intersection_ray.direction_y = 0.0;
        intersection_ray.direction_z = 0.0;

        // Initialize the untransformed a and b vectors.
        Vector3f a_vec;
        Vector3f b_vec;

        a_vec << camera_ray.direction_x, camera_ray.direction_y,
                 camera_ray.direction_z;
        b_vec << camera_ray.origin_x, camera_ray.origin_y,
                 camera_ray.origin_z;

        // Initial guess with quadratic equation.

        /* Inverse transform the a and b vectors to the origin coordinates of
         * the bounding unit sphere of the primitive.
         */
        Vector4f a_vec_4d;
        Vector4f b_vec_4d;

        a_vec_4d << a_vec(0), a_vec(1), a_vec(2), 1;
        b_vec_4d << b_vec(0), b_vec(1), b_vec(2), 1;

        Matrix4f inverse_transform_a =
            getTotalTransform(transformation_stack, true, false, false, true,
                              true);
        Matrix4f inverse_transform_b =
            getTotalTransform(transformation_stack, true, false, true, true,
                              true);

        a_vec_4d = inverse_transform_a * a_vec_4d;
        b_vec_4d = inverse_transform_b * b_vec_4d;

        Vector3f a_vec_3d;
        Vector3f b_vec_3d;

        a_vec_3d << a_vec_4d(0) / prm->getCoeff()(0),
                    a_vec_4d(1) / prm->getCoeff()(1),
                    a_vec_4d(2) / prm->getCoeff()(2);
        b_vec_3d << b_vec_4d(0) / prm->getCoeff()(0),
                    b_vec_4d(1) / prm->getCoeff()(1),
                    b_vec_4d(2) / prm->getCoeff()(2);

        float a = a_vec_3d.dot(a_vec_3d);
        float b = 2.0 * a_vec_3d.dot(b_vec_3d);
        float c = b_vec_3d.dot(b_vec_3d) - 3;

        float discriminant = b * b - 4.0 * a * c;
        if (discriminant < 0) {
            return intersection_ray;
        }

        /* For all the scenes provided, t_minus (with the discriminant
         * subtracted from -b in the quadratic formula) always corresponds to
         * the initial guess of the intersection point closer to the camera.
         */
        float t_minus = (-b - sqrt(discriminant)) / (2.0 * a);

        if (t_minus < 0) {
            return intersection_ray;
        }

        // Newton's method below.
        float t_old = t_minus;
        float t_new = t_old;
        float g_t_old = 10;

        // Stop Newton's method after 1000 iterations
        int iteration_count = 0;
        int max_iterations = 1000;

        Matrix4f inverse_transform =
            getTotalTransform(transformation_stack, true, false, true, true,
                              true);

        float limit = 0.0001;

        while (abs(g_t_old) >= limit && iteration_count < max_iterations) {
            t_old = t_new;
            Vector3f ray_pos = a_vec * t_old + b_vec;

            // Inverse transforming ray position with current t_old.
            Vector4f t_ray_pos;
            t_ray_pos << ray_pos(0), ray_pos(1), ray_pos(2), 1;

            t_ray_pos = inverse_transform * t_ray_pos;

            float t_x = t_ray_pos(0) / prm->getCoeff()(0);
            float t_y = t_ray_pos(1) / prm->getCoeff()(1);
            float t_z = t_ray_pos(2) / prm->getCoeff()(2);

            float e_val = prm->getExp0();
            float n_val = prm->getExp1();

            // g(t) = superquadric inside-outside function value
            g_t_old =
                pow(pow(pow(t_x, 2.0), 1.0 / e_val)
                    + pow(pow(t_y, 2.0), 1.0 / e_val), e_val / n_val)
                + pow(pow(t_z, 2.0), 1.0 / n_val) - 1.0;

            // g'(t) = a_vec_3d (dot) grad_S;
            float partial_Sx =
                (1.0 / n_val)
                * (2 * t_x * pow(pow(t_x, 2.0), (1.0 / e_val) - 1.0)
                   * pow(pow(pow(t_x, 2.0), 1.0 / e_val)
                   + pow(pow(t_y, 2.0), 1.0 / e_val),
                   (e_val / n_val) - 1.0));
            float partial_Sy =
                (1.0 / n_val)
                * (2 * t_y * pow(pow(t_y, 2.0), (1.0 / e_val) - 1.0)
                   * pow(pow(pow(t_x, 2.0), 1.0 / e_val)
                   + pow(pow(t_y, 2.0), 1.0 / e_val),
                   (e_val / n_val) - 1.0));
            float partial_Sz =
                (1.0 / n_val)
                * (2 * t_z * pow(pow(t_z, 2.0), (1.0 / n_val) - 1.0));

            Vector3f grad_S;
            grad_S << partial_Sx, partial_Sy, partial_Sz;

            float g_prime_t_old = a_vec_3d.dot(grad_S);

            // Check if g'(t) is equal to 0.
            if (g_prime_t_old == 0) {
                if (abs(g_t_old) < limit) {
                    break;
                }
                else {
                    return intersection_ray;
                }
            }

            // Check if g'(t) is positive.
            if (g_prime_t_old > 0) {
                if (abs(g_t_old) < limit) {
                    break;
                }
                else {
                    return intersection_ray;
                }
            }

            // Iterate t.
            t_new = t_old - (g_t_old / g_prime_t_old);

            iteration_count++;
        }

        if (abs(g_t_old) < limit) {
            intersection_ray.is_intersect = true;
        }

        /* Transform from the origin coordinates of the primitive to the
         * original world space coordinates to get the intersection point.
         */
        Vector3f iray_origin_itransformed = a_vec_3d * t_old + b_vec_3d;
        Matrix4f origin_transform =
            getTotalTransform(transformation_stack, false, false, true, true,
                              true);
        Vector4f iray_origin_transformed_4d;
        iray_origin_transformed_4d << iray_origin_itransformed(0)
                                      * prm->getCoeff()(0),
                                      iray_origin_itransformed(1)
                                      * prm->getCoeff()(1),
                                      iray_origin_itransformed(2)
                                      * prm->getCoeff()(2),
                                      1;
        iray_origin_transformed_4d =
            origin_transform * iray_origin_transformed_4d;

        Vector3f iray_origin_transformed_3d;
        iray_origin_transformed_3d << iray_origin_transformed_4d(0),
                                      iray_origin_transformed_4d(1),
                                      iray_origin_transformed_4d(2);

        intersection_ray.origin_x = iray_origin_transformed_3d(0);
        intersection_ray.origin_y = iray_origin_transformed_3d(1);
        intersection_ray.origin_z = iray_origin_transformed_3d(2);

        /* Transform the the origin coordinates of the primitive by the primitve
         * coeff values to obtain the normal vector of the corresponding
         * intersection point of the primitve's surface.
         */
        Vector4f iray_origin_scaled_4d;
        iray_origin_scaled_4d << iray_origin_itransformed(0)
                                 * prm->getCoeff()(0),
                                 iray_origin_itransformed(1)
                                 * prm->getCoeff()(1),
                                 iray_origin_itransformed(2)
                                 * prm->getCoeff()(2),
                                 1;

        Vector3f iray_origin_scaled_3d;
        iray_origin_scaled_3d << iray_origin_scaled_4d(0),
                                 iray_origin_scaled_4d(1),
                                 iray_origin_scaled_4d(2);

        Vector3f normal_vector = prm->getNormal(iray_origin_scaled_3d);

        // Use inverse transpose transformation for normal vector
        Matrix4f normal_transform =
            getTotalTransform(transformation_stack, false, true, false, true,
                              true);
        Vector4f normal4d;
        normal4d << normal_vector(0), normal_vector(1), normal_vector(2), 1;

        normal4d = normal_transform * normal4d;

        Vector3f normal3d;
        normal3d << normal4d(0), normal4d(1), normal4d(2);
        normal3d.normalize();

        intersection_ray.direction_x = normal3d(0);
        intersection_ray.direction_y = normal3d(1);
        intersection_ray.direction_z = normal3d(2);

        return intersection_ray;
    }

    /**
     * Given a renderable, return a vector of all the intersection rays of the
     * respective primitives contained by the renderable (or the renderable
     * itself if the renderable is a primitive).
     */
    void findAllIntersections(vector<Ray> &intersection_rays,
                              Ray camera_ray, Renderable *ren,
                              vector<Transformation> &transformation_stack) {
        if (ren->getType() == PRM) {
            Primitive *prm = dynamic_cast<Primitive*>(ren);

            // Get the intersection ray of the current Primitive.
            intersection_rays.push_back(findIntersection(prm,
                                                         transformation_stack,
                                                         camera_ray));

        } else if (ren->getType() == OBJ) {
            Object *obj = dynamic_cast<Object*>(ren);
            const vector<Transformation>& overall_trans =
                obj->getOverallTransformation();
            for (int i = overall_trans.size() - 1; i >= 0; i--) {
                transformation_stack.push_back(overall_trans.at(i));
            }

            for (auto& child_it : obj->getChildren()) {
                const vector<Transformation>& child_trans =
                    child_it.second.transformations;
                for (int i = child_trans.size() - 1; i >= 0; i--) {
                    transformation_stack.push_back(child_trans.at(i));
                }
                findAllIntersections(
                    intersection_rays, camera_ray,
                    Renderable::get(child_it.second.name),
                    transformation_stack);
                transformation_stack.erase(
                    transformation_stack.end() - child_trans.size(),
                    transformation_stack.end());
            }

            transformation_stack.erase(
                transformation_stack.end() - overall_trans.size(),
                transformation_stack.end());
        } else {
            fprintf(stderr, "Renderer::draw ERROR invalid RenderableType %d\n",
                ren->getType());
            exit(1);
        }
    }

    void drawIntersectTest(Camera *camera) {
        Ray camera_ray;
        camera_ray.origin_x = camera->getPosition()(0);
        camera_ray.origin_y = camera->getPosition()(1);
        camera_ray.origin_z = camera->getPosition()(2);

        Vector4f a_vec;
        a_vec << 0, 0, -1, 1;

        // Rotate a_vec by camera rotation transformation.
        Matrix4f camera_rotation;
        float rx = camera->getAxis()(0);
        float ry = camera->getAxis()(1);
        float rz = camera->getAxis()(2);
        float angle = camera->getAngle();

        float rotation_axis_magnitude = sqrt(rx * rx + ry * ry + rz * rz);
        rx /= rotation_axis_magnitude;
        ry /= rotation_axis_magnitude;
        rz /= rotation_axis_magnitude;

        float a00 = rx * rx + (1 - rx * rx) * cos(angle);
        float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
        float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
        float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
        float a11 = ry * ry + (1 - ry * ry) * cos(angle);
        float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
        float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
        float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
        float a22 = rz * rz + (1 - rz * rz) * cos(angle);

        camera_rotation <<
            a00, a01, a02, 0,
            a10, a11, a12, 0,
            a20, a21, a22, 0,
            0, 0, 0, 1;

        a_vec = camera_rotation * a_vec;

        camera_ray.direction_x = a_vec(0);
        camera_ray.direction_y = a_vec(1);
        camera_ray.direction_z = a_vec(2);

        // Get the currently selected renderable.
        const Line* cur_state = CommandLine::getState();
        Renderable* current_selected_ren = NULL;

        if (cur_state) {
            current_selected_ren = Renderable::get(cur_state->tokens[1]);
        }

        if (current_selected_ren == NULL
            || current_selected_ren->getType() == PRM) {
            return;
        }

        // Find all intersection rays of the primitives.
        vector<Ray> intersection_rays;
        vector<Transformation> transformation_stack;
        findAllIntersections(intersection_rays, camera_ray,
                             current_selected_ren, transformation_stack);

        float min_distance = 1000000;
        Ray closest_intersection_ray;
        bool found_intersection = false;
        for (int i = 0; i < (int) intersection_rays.size(); i++) {
            Ray intersection_ray = intersection_rays[i];

            /* Check if the current intersection ray is the closest to the
             * camera, and update the closest intersection ray if needed.
             */
            float iray_x = intersection_ray.origin_x;
            float iray_y = intersection_ray.origin_y;
            float iray_z = intersection_ray.origin_z;

            float distance = sqrt(pow(iray_x - camera_ray.origin_x, 2)
                                  + pow(iray_y - camera_ray.origin_y, 2)
                                  + pow(iray_z - camera_ray.origin_z, 2));

            if (distance < min_distance && intersection_ray.is_intersect) {
                min_distance = distance;
                closest_intersection_ray = intersection_ray;
                found_intersection = true;
            }
        }

        // Draw the closest intersection ray if there is one.
        if (found_intersection) {
            const float IO_test_color[3] = {0.5, 0.2, 0.7};
            glMaterialfv(GL_FRONT, GL_AMBIENT, IO_test_color);
            glLineWidth(10.0);
            glBegin(GL_LINES);
            glVertex3f(
                closest_intersection_ray.origin_x,
                closest_intersection_ray.origin_y,
                closest_intersection_ray.origin_z);
            Vector3f endpoint = closest_intersection_ray.getLocation(1.0);
            glVertex3f(endpoint[0], endpoint[1], endpoint[2]);
            glEnd();
        }
    }

    /* Assignment Part B */

    /* Ray traces the scene. */
    void raytrace(Camera camera, Scene scene) {
        // LEAVE THIS UNLESS YOU WANT TO WRITE YOUR OWN OUTPUT FUNCTION
        PNGMaker png = PNGMaker(XRES, YRES);

        // REPLACE THIS WITH YOUR CODE
        for (int i = 0; i < XRES; i++) {
            for (int j = 0; j < YRES; j++) {
                png.setPixel(i, j, 1.0, 1.0, 1.0);
            }
        }

        // LEAVE THIS UNLESS YOU WANT TO WRITE YOUR OWN OUTPUT FUNCTION
        if (png.saveImage()) {
            fprintf(stderr, "Error: couldn't save PNG image\n");
        } else {
            printf("DONE!\n");
        }
    }
};
