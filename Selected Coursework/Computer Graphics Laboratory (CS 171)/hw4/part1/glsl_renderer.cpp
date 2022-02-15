// Philip Carr
// CS 171 Assignment 4 Part 1
// November 17, 2018
// glsl_renderer.cpp

#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <math.h>
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>

#include "scene_read.h"
#include "quaternion.h"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// quaternion class member function definitions.

// Default constructor for a quaternion.
quaternion::quaternion() {
    s = 0;
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
}

// 4-argument constructor for a quaternion.
quaternion::quaternion(const float s, const float v0, const float v1,
                       const float v2) {
    this->s = s;
    this->v[0] = v0;
    this->v[1] = v1;
    this->v[2] = v2;
}

// Return the quaternion's s-value.
float quaternion::get_s() const {
    return this->s;
}

// Return the quaternion's v-vector.
float *quaternion::get_v() {
    return this->v;
}

// Set the value of s of a quaternion.
void quaternion::set_s(const float s) {
    this->s = s;
}

// Set the vector v of a quaternion.
void quaternion::set_v(const float v0, const float v1, const float v2) {
    this->v[0] = v0;
    this->v[1] = v1;
    this->v[2] = v2;
}

// Add a quaternion q to the given quaternion.
void quaternion::add(quaternion &q) {
    float qs = q.get_s();
    float *qv = q.get_v();
    float qx = qv[0];
    float qy = qv[1];
    float qz = qv[2];

    this->s += qs;
    this->v[0] += qx;
    this->v[1] += qy;
    this->v[2] += qz;
}

// Multiply a scalar s to the given quaternion.
void quaternion::s_mult(const float s) {
    this->s *= s;
    this->v[0] *= s;
    this->v[1] *= s;
    this->v[2] *= s;
}

// Return the product of the quaternion q and the given quaternion.
quaternion quaternion::q_mult(quaternion &q) const { // Use Eigen
    float qs = q.get_s();
    float *qv = q.get_v();
    float qx = qv[0];
    float qy = qv[1];
    float qz = qv[2];

    float mult_s = this->s * qs - this->v[0] * qx - this->v[1] * qy
                   - this->v[2] * qz;
    float mult_v0 = this->s * qx + this->v[0] * qs + this->v[1] * qz
                    - this->v[2] * qy;
    float mult_v1 = this->s * qy + this->v[1] * qs + this->v[2] * qx
                    - this->v[0] * qz;
    float mult_v2 = this->s * qz + this->v[2] * qs + this->v[0] * qy
                    - this->v[1] * qx;

    return quaternion(mult_s, mult_v0, mult_v1, mult_v2);
}

// Return the complex conjugate of the given quaternion.
quaternion quaternion::conj() const {
    float neg_v0 = -this->v[0];
    float neg_v1 = -this->v[1];
    float neg_v2 = -this->v[2];
    return quaternion(this->s, neg_v0, neg_v1, neg_v2);
}

// Return the norm of the given quaternion (sqrt(s^2 + x^2 + y^2 + z^2)).
float quaternion::norm() const {
    return sqrt(this->s * this->s + this->v[0] * this->v[0]
           + this->v[1] * this->v[1] + this->v[2] * this->v[2]);
}

// Return the inverse of the given quaternion (q^*/(|q|^2)).
quaternion quaternion::inv() const {
    float norm = this->norm();
    quaternion q_star_normalized = this->conj();
    q_star_normalized.set_s(q_star_normalized.get_s() / (norm * norm));
    float *v = q_star_normalized.get_v();
    q_star_normalized.set_v(v[0] / (norm * norm), v[1] / (norm * norm),
                            v[2] / (norm * norm));
    return q_star_normalized;
}

// Normalize the given quaternion.
void quaternion::normalize() {
    float norm = this->norm();
    this->s = this->s / norm;
    this->v[0] = this->v[0] / norm;
    this->v[1] = this->v[1] / norm;
    this->v[2] = this->v[2] / norm;
}

////////////////////////////////////////////////////////////////////////////////

void init(void);
void readShaders();
void reshape(int width, int height);
void display(void);

void init_lights();
void set_lights();
void draw_objects();

void mouse_pressed(int button, int state, int x, int y);
void mouse_moved(int x, int y);
void key_pressed(unsigned char key, int x, int y);

void update_rotations(int x, int y);

////////////////////////////////////////////////////////////////////////////////

// Camera and lists of lights and objects.

Camera cam;
vector<Point_Light> lights;
vector<Object> objects;

////////////////////////////////////////////////////////////////////////////////

/* The following are parameters for creating an interactive first-person camera
 * view of the scene. The variables will make more sense when explained in
 * context, so you should just look at the 'mousePressed', 'mouseMoved', and
 * 'keyPressed' functions for the details.
 */

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

const float step_size = 0.2;
const float x_view_step = 90.0, y_view_step = 90.0;
float x_view_angle = 0, y_view_angle = 0;

quaternion last_rotation;
quaternion current_rotation;

bool is_pressed = false;
bool wireframe_mode = false;

int xres;
int yres;

GLenum shaderProgram;
string vertProgFileName, fragProgFileName;
GLint n_lights;

int mode;

////////////////////////////////////////////////////////////////////////////////

/* Initialze OpenGL system and organize scene data (camera, lights, and
 * objects).
 */
void init(string filename, int mode)
{
    lights = read_lights(filename);

    /* Check mode value here: if mode == 0, use Gouraud Shading. If mode == 1,
     * use Phong shaders with the readShaders function.
     */
    if (mode == 0) {
        glShadeModel(GL_SMOOTH);
    }
    else {
        readShaders();
    }

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_NORMALIZE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);


    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    cam = read_camera(filename);

    glFrustum(cam.l, cam.r, cam.b, cam.t, cam.n, cam.f);

    glMatrixMode(GL_MODELVIEW);

    objects = read_objects(filename);

    init_lights();

    last_rotation = quaternion(1, 0, 0, 0);
    current_rotation = quaternion(1, 0, 0, 0);
}

/**
 * Read glsl vertex shader and fragment shader files and compile them together
 * to create a shader program.
 */
void readShaders() {
   string vertProgramSource, fragProgramSource;

   ifstream vertProgFile(vertProgFileName.c_str());
   if (! vertProgFile)
      cerr << "Error opening vertex shader program\n";
   ifstream fragProgFile(fragProgFileName.c_str());
   if (! fragProgFile)
      cerr << "Error opening fragment shader program\n";

   getline(vertProgFile, vertProgramSource, '\0');
   const char* vertShaderSource = vertProgramSource.c_str();

   getline(fragProgFile, fragProgramSource, '\0');
   const char* fragShaderSource = fragProgramSource.c_str();

   char buf[1024];
   GLsizei blah;

   // Initialize shaders
   GLenum vertShader, fragShader;

   shaderProgram = glCreateProgram();

   vertShader = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(vertShader, 1, &vertShaderSource, NULL);
   glCompileShader(vertShader);

   GLint isCompiled = 0;
   glGetShaderiv(vertShader, GL_COMPILE_STATUS, &isCompiled);
   if(isCompiled == GL_FALSE)
   {
      GLint maxLength = 0;
      glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &maxLength);

      // The maxLength includes the NULL character
      std::vector<GLchar> errorLog(maxLength);
      glGetShaderInfoLog(vertShader, maxLength, &maxLength, &errorLog[0]);

      // Provide the infolog in whatever manor you deem best.
      // Exit with failure.
      for (int i = 0; i < errorLog.size(); i++)
         cout << errorLog[i];
      glDeleteShader(vertShader); // Don't leak the shader.
      return;
   }

   fragShader = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(fragShader, 1, &fragShaderSource, NULL);
   glCompileShader(fragShader);

   isCompiled = 0;
   glGetShaderiv(fragShader, GL_COMPILE_STATUS, &isCompiled);
   if(isCompiled == GL_FALSE)
   {
      GLint maxLength = 0;
      glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &maxLength);

      // The maxLength includes the NULL character
      std::vector<GLchar> errorLog(maxLength);
      glGetShaderInfoLog(fragShader, maxLength, &maxLength, &errorLog[0]);

      // Provide the infolog in whatever manor you deem best.
      // Exit with failure.
      for (int i = 0; i < errorLog.size(); i++)
         cout << errorLog[i];
      glDeleteShader(fragShader); // Don't leak the shader.
      return;
   }

   glAttachShader(shaderProgram, vertShader);
   glAttachShader(shaderProgram, fragShader);
   glLinkProgram(shaderProgram);
   cerr << "Enabling fragment program: " << gluErrorString(glGetError()) << endl;
   glGetProgramInfoLog(shaderProgram, 1024, &blah, buf);
   cerr << buf;

   cerr << "Enabling program object" << endl;
   glUseProgram(shaderProgram);

   // Pass the total number of lights into the shader program.
   n_lights = glGetUniformLocation(shaderProgram, "n_lights");
   glUniform1i(n_lights, (int) lights.size());
}

/**
 * Reshape the image whenever the window size changes.
 */
void reshape(int width, int height)
{
    height = (height == 0) ? 1 : height;
    width = (width == 0) ? 1 : width;

    glViewport(0, 0, width, height);

    mouse_scale_x = (float) (cam.r - cam.l) / (float) width;
    mouse_scale_y = (float) (cam.t - cam.b) / (float) height;

    glutPostRedisplay();
}

/**
 * Display the scene using OpenGL.
 */
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    float ori_mag = sqrt(cam.ori_axis[0] * cam.ori_axis[0]
                         + cam.ori_axis[1] * cam.ori_axis[1]
                         + cam.ori_axis[2] * cam.ori_axis[2]);
    glRotatef(-cam.ori_angle * 180 / M_PI,
              cam.ori_axis[0] / ori_mag, cam.ori_axis[1] / ori_mag,
              cam.ori_axis[2] / ori_mag);

    glTranslatef(-cam.pos[0], -cam.pos[1], -cam.pos[2]);

    quaternion rotation = current_rotation.q_mult(last_rotation);
    float qs = rotation.get_s();
    float *v = rotation.get_v();
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

    set_lights();

    draw_objects();

    glutSwapBuffers();
}

/**
 * Initialze the lights of the scene.
 */
void init_lights()
{

    glEnable(GL_LIGHTING);

    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {

        int light_id = GL_LIGHT0 + i;

        glEnable(light_id);

        glLightfv(light_id, GL_AMBIENT, lights[i].color);
        glLightfv(light_id, GL_DIFFUSE, lights[i].color);
        glLightfv(light_id, GL_SPECULAR, lights[i].color);

        glLightf(light_id, GL_QUADRATIC_ATTENUATION, lights[i].k);
    }
}

/**
 * Set the lights of the scene.
 */
void set_lights()
{
    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {
        int light_id = GL_LIGHT0 + i;

        glLightfv(light_id, GL_POSITION, lights[i].pos);
    }
}

/**
 * Draw the objects in the scene.
 */
void draw_objects()
{
    int num_objects = objects.size();

    for(int i = 0; i < num_objects; ++i)
    {
        glPushMatrix();

        {
            int num_transform_sets = objects[i].transform_sets.size();

            /* Modify the current modelview matrix with the
             * geometric transformations for this object.
             */
            for(int j = num_transform_sets - 1; j >= 0 ; j--) {
                Transform transform = objects[i].transform_sets[j];
                if (transform.type == "t") {
                    glTranslatef(transform.components[0],
                                 transform.components[1],
                                 transform.components[2]);
                }
                else if (transform.type == "r") {
                    glRotatef(transform.rotation_angle,
                              transform.components[0], transform.components[1],
                              transform.components[2]);
                }
                else {
                    glScalef(transform.components[0], transform.components[1],
                             transform.components[2]);
                }
            }

            glMaterialfv(GL_FRONT, GL_AMBIENT, objects[i].ambient_reflect);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, objects[i].diffuse_reflect);
            glMaterialfv(GL_FRONT, GL_SPECULAR, objects[i].specular_reflect);
            glMaterialf(GL_FRONT, GL_SHININESS, objects[i].shininess);

            glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);

            glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);

            int buffer_size = objects[i].vertex_buffer.size();

            if(!wireframe_mode)
                glDrawArrays(GL_TRIANGLES, 0, buffer_size);
            else
                for(int j = 0; j < buffer_size; j += 3)
                    glDrawArrays(GL_LINE_LOOP, j, 3);
        }

        glPopMatrix();
    }
}

/**
 * Function to tell OpenGL what to do when the (left) mouse (button) is pressed.
 */
void mouse_pressed(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        mouse_x = x;
        mouse_y = y;

        is_pressed = true;
    }
    else if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
    {
        is_pressed = false;

        last_rotation = current_rotation.q_mult(last_rotation);
        current_rotation = quaternion(1, 0, 0, 0);
    }
}

/**
 * Update the quaternion rotations current_rotation and last_rotation using the
 * Arcball algorithm.
 */
void update_rotations(int x, int y) {
    float x1 = (mouse_x - ((float (xres - 1)) / 2)) / ((float) xres / 2);
    float y1 = -(mouse_y - ((float (yres - 1)) / 2)) / ((float) yres / 2);
    float z1;
    if (x1 * x1 + y1 * y1 <= 1) {
        z1 = sqrt(1 - x1 * x1 - y1 * y1);
    }
    else {
        z1 = (float) 0;
    }

    float x2 = (x - ((float (xres - 1)) / 2)) / ((float) xres / 2);
    float y2 = -(y - ((float (yres - 1)) / 2)) / ((float) yres / 2);
    float z2;
    if (x2 * x2 + y2 * y2 <= 1) {
        z2 = sqrt(1 - x2 * x2 - y2 * y2);
    }
    else {
        z2 = (float) 0;
    }

    float v1_norm = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    float v2_norm = sqrt(x2 * x2 + y2 * y2 + z2 * z2);

    float theta = acos(min((float) 1, (x1 * x2 + y1 * y2 + z1 * z2)
                                      / (v1_norm * v2_norm)));

    float u[3] = {y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2};
    float u_magnitude = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (u_magnitude != 0) {
        u[0] = u[0] / u_magnitude;
        u[1] = u[1] / u_magnitude;
        u[2] = u[2] / u_magnitude;
    }

    float qs = cos(theta / 2);
    float qx = u[0] * sin(theta / 2);
    float qy = u[1] * sin(theta / 2);
    float qz = u[2] * sin(theta / 2);

    current_rotation = quaternion(qs, qx, qy, qz);
    current_rotation.normalize();

    last_rotation = current_rotation.q_mult(last_rotation);
}

/**
 * Tells OpenGL what to do when the mouse is moved.
 */
void mouse_moved(int x, int y)
{
    if(is_pressed)
    {
        update_rotations(x, y);

        mouse_x = x;
        mouse_y = y;

        glutPostRedisplay();
    }
}

/* 'deg2rad' function:
 *
 * Converts given angle in degrees to radians.
 */
float deg2rad(float angle)
{
    return angle * M_PI / 180.0;
}

/**
 * Function tells OpenGL what to do when a key is pressed.
 */
void key_pressed(unsigned char key, int x, int y)
{

    if(key == 'q')
    {
        exit(0);
    }

    else if(key == 't')
    {
        wireframe_mode = !wireframe_mode;

        glutPostRedisplay();
    }
    else
    {

        float x_view_rad = deg2rad(x_view_angle);

        /* 'w' for step forward
         */
        if(key == 'w')
        {
            cam.pos[0] += step_size * sin(x_view_rad);
            cam.pos[2] -= step_size * cos(x_view_rad);
            glutPostRedisplay();
        }
        /* 'a' for step left
         */
        else if(key == 'a')
        {
            cam.pos[0] -= step_size * cos(x_view_rad);
            cam.pos[2] -= step_size * sin(x_view_rad);
            glutPostRedisplay();
        }
        /* 's' for step backward
         */
        else if(key == 's')
        {
            cam.pos[0] -= step_size * sin(x_view_rad);
            cam.pos[2] += step_size * cos(x_view_rad);
            glutPostRedisplay();
        }
        /* 'd' for step right
         */
        else if(key == 'd')
        {
            cam.pos[0] += step_size * cos(x_view_rad);
            cam.pos[2] += step_size * sin(x_view_rad);
            glutPostRedisplay();
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
    if (argc != 5) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [scene_description_file.txt] [xres]"
                                 + " [yres] [mode]";
        cout << usage_statement << endl;
        return 1;
    }

    xres = atoi(argv[2]);
    yres = atoi(argv[3]);
    mode = atoi(argv[4]);

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(xres, yres);

    glutInitWindowPosition(0, 0);

    glutCreateWindow("GLSL Test");

    vertProgFileName = "vertexProgram.glsl";
    fragProgFileName = "fragmentProgram.glsl";
    init(argv[1], mode);

    glutDisplayFunc(display);

    glutReshapeFunc(reshape);

    glutMouseFunc(mouse_pressed);

    glutMotionFunc(mouse_moved);

    glutKeyboardFunc(key_pressed);

    glutMainLoop();
}
