// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// keyframe.cpp

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <math.h>
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>

#include "script_read.h"

void init(void);
void reshape(int width, int height);
void display(void);

void init_camera();
void drawIBar();

void key_pressed(unsigned char key, int x, int y);

void update_rotations(int x, int y);

////////////////////////////////////////////////////////////////////////////////

// Camera, Script_Data, and I_Bar.
Camera cam;

Script_Data script_data;
int current_frame_number;

/* Needed to draw the cylinders using glu */
GLUquadricObj *quadratic;

Vector3f interpolated_translation;
Vector3f interpolated_scale;
quaternion interpolated_rotation;

////////////////////////////////////////////////////////////////////////////////

/* The following are parameters for creating a first-person camera view of the
 * scene.
 */

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

int xres;
int yres;

////////////////////////////////////////////////////////////////////////////////

/* Initialze OpenGL system and organize scene data (camera, lights, and
 * objects).
 */
void init(string filename) {

    glShadeModel(GL_SMOOTH);

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_NORMALIZE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    init_camera();

    script_data = read_script_file(filename);

    glFrustum(cam.l, cam.r, cam.b, cam.t, cam.n, cam.f);

    glMatrixMode(GL_MODELVIEW);

    quadratic = gluNewQuadric();

    current_frame_number = 0;

    interpolated_translation << 0, 0, 0;
    interpolated_scale << 1, 1, 1;
    interpolated_rotation = quaternion(1, 0, 0, 0);
}

/**
 * Reshape the image whenever the window size changes.
 */
void reshape(int width, int height) {
    height = (height == 0) ? 1 : height;
    width = (width == 0) ? 1 : width;

    glViewport(0, 0, width, height);

    glutPostRedisplay();
}

/**
 * Display the scene using OpenGL.
 */
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    glTranslatef(-cam.pos[0], -cam.pos[1], -cam.pos[2]);

    glTranslatef(interpolated_translation(0), interpolated_translation(1),
                 interpolated_translation(2));

    glScalef(interpolated_scale(0), interpolated_scale(1),
             interpolated_scale(2));

    float qs = interpolated_rotation.get_s();
    float *v = interpolated_rotation.get_v();
    float qx = v[0];
    float qy = v[1];
    float qz = v[2];
    float rotation_matrix[16] = {
        1 - 2 * qy * qy - 2 * qz * qz, 2 * (qx * qy + qz * qs),
        2 * (qx * qz - qy * qs), 0,

        2 * (qx * qy - qz * qs), 1 - 2 * qx * qx - 2 * qz * qz,
        2 * (qy * qz + qx * qs), 0,

        2 * (qx * qz + qy * qs), 2 * (qy * qz - qx * qs),
        1 - 2 * qx * qx - 2 * qy * qy, 0,

        0, 0, 0, 1
    };

    glMultMatrixf(rotation_matrix);

    drawIBar();

    glutSwapBuffers();
}

/**
 * Initialze the camera.
 */
void init_camera() {
    cam.n = 1.0;
    cam.f = 60.0;
    cam.l = -1.0;
    cam.r = 1.0;
    cam.b = -1.0;
    cam.t = 1.0;

    cam.pos[0] = 0;
    cam.pos[1] = 0;
    cam.pos[2] = 40;

    cam.ori_angle = 0;

    cam.ori_axis[0] = 1.0;
    cam.ori_axis[1] = 0.0;
    cam.ori_axis[2] = 0.0;
}

/**
 * Draw the I_Bar object in the scene.
 */
void drawIBar() {
    /* Parameters for drawing the cylinders */
    float cyRad = 0.2, cyHeight = 1.0;
    int quadStacks = 4, quadSlices = 4;

    glPushMatrix();
    glColor3f(0, 0, 1);
    glTranslatef(0, cyHeight, 0);
    glRotatef(90, 1, 0, 0);
    gluCylinder(quadratic, cyRad, cyRad, 2.0 * cyHeight, quadSlices, quadStacks);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0, 1, 1);
    glTranslatef(0, cyHeight, 0);
    glRotatef(90, 0, 1, 0);
    gluCylinder(quadratic, cyRad, cyRad, cyHeight, quadSlices, quadStacks);
    glPopMatrix();

    glPushMatrix();
    glColor3f(1, 0, 1);
    glTranslatef(0, cyHeight, 0);
    glRotatef(-90, 0, 1, 0);
    gluCylinder(quadratic, cyRad, cyRad, cyHeight, quadSlices, quadStacks);
    glPopMatrix();

    glPushMatrix();
    glColor3f(1, 1, 0);
    glTranslatef(0, -cyHeight, 0);
    glRotatef(-90, 0, 1, 0);
    gluCylinder(quadratic, cyRad, cyRad, cyHeight, quadSlices, quadStacks);
    glPopMatrix();

    glPushMatrix();
    glColor3f(0, 1, 0);
    glTranslatef(0, -cyHeight, 0);
    glRotatef(90, 0, 1, 0);
    gluCylinder(quadratic, cyRad, cyRad, cyHeight, quadSlices, quadStacks);
    glPopMatrix();
}

/* 'deg2rad' function:
 *
 * Converts given angle in degrees to radians.
 */
float deg2rad(float angle) {
    return angle * M_PI / 180.0;
}

/* 'rad2deg' function:
 *
 * Converts given angle in radians to degrees.
 */
float rad2deg(float angle) {
    return angle * 180.0 / M_PI;
}

/**
 * Return the four keyframes used for interpolation given the global variable
 * current_frame_number.
 */
vector<Keyframe> get_interpolation_keyframes() {
    Keyframe kf_before_start, kf_start, kf_end, kf_after_end;
    int i_kf_start, i_kf_end;

    vector<Keyframe> keyframes = script_data.keyframes;

    for (int i = 0; i < (int) keyframes.size(); i++) {
        int frame_number = keyframes[i].frame_number;
        if (frame_number <= current_frame_number) {
            kf_start = keyframes[i];
            i_kf_start = i;
        }
        else {
            break;
        }
    }

    if (kf_start.frame_number
        == keyframes[(int) keyframes.size() - 1].frame_number) {
        kf_end = keyframes[0];
        i_kf_end = 0;
    }
    else {
        kf_end = keyframes[i_kf_start+1];
        i_kf_end = i_kf_start+1;
    }

    if (kf_end.frame_number
        == keyframes[(int) keyframes.size() - 1].frame_number) {
        kf_after_end = keyframes[0];
    }
    else {
        kf_after_end = keyframes[i_kf_end+1];
    }

    if (kf_start.frame_number == keyframes[0].frame_number) {
        kf_before_start = keyframes[(int) keyframes.size() - 1];
    }
    else {
        kf_before_start = keyframes[i_kf_start-1];
    }

    vector<Keyframe> interpolation_keyframes;
    interpolation_keyframes.push_back(kf_before_start);
    interpolation_keyframes.push_back(kf_start);
    interpolation_keyframes.push_back(kf_end);
    interpolation_keyframes.push_back(kf_after_end);

    return interpolation_keyframes;
}

/**
 * Return the basis matrix used for interpolation given tension t. Catmull-Rom
 * splines use t = 0.
 */
Matrix4f get_basis_matrix(float t) {
    float s = 0.5 * (1 - t);
    Matrix4f B;
    B << 0, 1, 0, 0,
         -s, 0, s, 0,
         2 * s, s - 3, 3 - 2 * s, -s,
         -s, 2 - s, s - 2, s;
    return B;
}

/**
 * Interpolate the translation of the current frame given the global variable
 * current_frame_number, and the two function arguments the keyframes used for
 * interpolation and the basis matrix B.
 */
void interpolate_translation(vector<Keyframe> interpolation_keyframes,
                             Matrix4f B) {
    float u;
    if (interpolation_keyframes[2].frame_number
        > interpolation_keyframes[1].frame_number) {
        u = (float) (current_frame_number - interpolation_keyframes[1].frame_number)
            / (interpolation_keyframes[2].frame_number
               - interpolation_keyframes[1].frame_number);
    }
    else {
        u = (float) (current_frame_number - interpolation_keyframes[1].frame_number)
            / (script_data.n_frames - interpolation_keyframes[1].frame_number);
    }

    Vector4f u_vector;
    u_vector << 1, u, u * u, u * u * u;

    Vector4f x_values;
    x_values << interpolation_keyframes[0].translation(0),
                interpolation_keyframes[1].translation(0),
                interpolation_keyframes[2].translation(0),
                interpolation_keyframes[3].translation(0);

    Vector4f y_values;
    y_values << interpolation_keyframes[0].translation(1),
                interpolation_keyframes[1].translation(1),
                interpolation_keyframes[2].translation(1),
                interpolation_keyframes[3].translation(1);

    Vector4f z_values;
    z_values << interpolation_keyframes[0].translation(2),
                interpolation_keyframes[1].translation(2),
                interpolation_keyframes[2].translation(2),
                interpolation_keyframes[3].translation(2);

    float interpolated_x = u_vector.dot(B * x_values);
    float interpolated_y = u_vector.dot(B * y_values);
    float interpolated_z = u_vector.dot(B * z_values);

    interpolated_translation << interpolated_x, interpolated_y, interpolated_z;
}

/**
 * Interpolate the scaling of the current frame given the global variable
 * current_frame_number, and the two function arguments the keyframes used for
 * interpolation and the basis matrix B.
 */
void interpolate_scale(vector<Keyframe> interpolation_keyframes,
                             Matrix4f B) {
    float u;
    if (interpolation_keyframes[2].frame_number
        > interpolation_keyframes[1].frame_number) {
        u = (float) (current_frame_number - interpolation_keyframes[1].frame_number)
            / (interpolation_keyframes[2].frame_number
               - interpolation_keyframes[1].frame_number);
    }
    else {
        u = (float) (current_frame_number - interpolation_keyframes[1].frame_number)
            / (script_data.n_frames - interpolation_keyframes[1].frame_number);
    }

    Vector4f u_vector;
    u_vector << 1, u, u * u, u * u * u;

    Vector4f x_values;
    x_values << interpolation_keyframes[0].scale(0),
                interpolation_keyframes[1].scale(0),
                interpolation_keyframes[2].scale(0),
                interpolation_keyframes[3].scale(0);

    Vector4f y_values;
    y_values << interpolation_keyframes[0].scale(1),
                interpolation_keyframes[1].scale(1),
                interpolation_keyframes[2].scale(1),
                interpolation_keyframes[3].scale(1);

    Vector4f z_values;
    z_values << interpolation_keyframes[0].scale(2),
                interpolation_keyframes[1].scale(2),
                interpolation_keyframes[2].scale(2),
                interpolation_keyframes[3].scale(2);

    float interpolated_x = u_vector.dot(B * x_values);
    float interpolated_y = u_vector.dot(B * y_values);
    float interpolated_z = u_vector.dot(B * z_values);

    interpolated_scale << interpolated_x, interpolated_y, interpolated_z;
}

/**
 * Interpolate the rotation of the current frame given the global variable
 * current_frame_number, and the two function arguments the keyframes used for
 * interpolation and the basis matrix B. Rotations are represented using
 * quaternions here.
 */
void interpolate_rotation(vector<Keyframe> interpolation_keyframes,
                             Matrix4f B) {
    float u;
    if (interpolation_keyframes[2].frame_number
        > interpolation_keyframes[1].frame_number) {
        u = (float) (current_frame_number - interpolation_keyframes[1].frame_number)
            / (interpolation_keyframes[2].frame_number
               - interpolation_keyframes[1].frame_number);
    }
    else {
        u = (float) (current_frame_number - interpolation_keyframes[1].frame_number)
            / (script_data.n_frames - interpolation_keyframes[1].frame_number);
    }

    Vector4f u_vector;
    u_vector << 1, u, u * u, u * u * u;

    Vector4f x_values;
    x_values << interpolation_keyframes[0].rotation.get_v()[0],
                interpolation_keyframes[1].rotation.get_v()[0],
                interpolation_keyframes[2].rotation.get_v()[0],
                interpolation_keyframes[3].rotation.get_v()[0];

    Vector4f y_values;
    y_values << interpolation_keyframes[0].rotation.get_v()[1],
                interpolation_keyframes[1].rotation.get_v()[1],
                interpolation_keyframes[2].rotation.get_v()[1],
                interpolation_keyframes[3].rotation.get_v()[1];

    Vector4f z_values;
    z_values << interpolation_keyframes[0].rotation.get_v()[2],
                interpolation_keyframes[1].rotation.get_v()[2],
                interpolation_keyframes[2].rotation.get_v()[2],
                interpolation_keyframes[3].rotation.get_v()[2];

    Vector4f s_values;
    s_values << interpolation_keyframes[0].rotation.get_s(),
                interpolation_keyframes[1].rotation.get_s(),
                interpolation_keyframes[2].rotation.get_s(),
                interpolation_keyframes[3].rotation.get_s();

    float interpolated_x = u_vector.dot(B * x_values);
    float interpolated_y = u_vector.dot(B * y_values);
    float interpolated_z = u_vector.dot(B * z_values);
    float interpolated_rotation_angle = u_vector.dot(B * s_values);

    quaternion new_rotation = quaternion(interpolated_rotation_angle,
                                       interpolated_x, interpolated_y,
                                       interpolated_z);
    new_rotation.normalize();

    interpolated_rotation = new_rotation;
}

/**
 * Function tells OpenGL what to do when a key is pressed.
 */
void key_pressed(unsigned char key, int x, int y) {

    if (key == 'q') {
        exit(0);
    }
    /* Interpolate the translation, scaling, and rotation of the current frame
     * given the global variable current_frame_number.
     */
    if (key == 'i') {
        vector<Keyframe> interpolation_keyframes =
            get_interpolation_keyframes();

        Matrix4f B = get_basis_matrix(0);

        interpolate_translation(interpolation_keyframes, B);
        interpolate_scale(interpolation_keyframes, B);
        interpolate_rotation(interpolation_keyframes, B);

        glutPostRedisplay();

        cout << "Frame " << current_frame_number << " rendered" << endl;

        current_frame_number++;
        if (current_frame_number == script_data.n_frames) {
            current_frame_number = 0;
        }
    }
}

/* The 'main' function:
 *
 * Run the OpenGL program (initialize OpenGL, display the scene, and react to
 * mouse and keyboard presses).
 */
int main(int argc, char* argv[])
{
    if (argc != 4) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [script_file.script] [xres]"
                                 + " [yres]";
        cout << usage_statement << endl;
        return 1;
    }

    xres = atoi(argv[2]);
    yres = atoi(argv[3]);

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(xres, yres);

    glutInitWindowPosition(0, 0);

    glutCreateWindow("GLSL Test");

    init(argv[1]);

    glutDisplayFunc(display);

    glutReshapeFunc(reshape);

    glutKeyboardFunc(key_pressed);

    glutMainLoop();
}
