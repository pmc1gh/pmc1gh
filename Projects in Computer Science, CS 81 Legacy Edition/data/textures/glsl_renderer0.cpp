// Philip Carr
// CS 81a Project: Illustrative Rendering
// November 18, 2019
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
using namespace Eigen;

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

void init(string texture_image_filename, string texture_normals_filename);
extern GLenum readpng(const char *filename);
void readShaders();
void reshape(int width, int height);
void display(void);

void create_lights();
void create_camera();
void create_square();

void init_lights();
void set_lights();
void draw_objects();

void mouse_pressed(int button, int state, int x, int y);
void mouse_moved(int x, int y);
void key_pressed(unsigned char key, int x, int y);

void update_rotations(int x, int y);

void transform_objects(bool init);

// cel shading functions
void draw_object_outlines();

////////////////////////////////////////////////////////////////////////////////

// Camera and lists of lights and objects.

Camera cam;
vector<Point_Light> lights;
vector<Object> objects;
vector<Object> original_objects;

Vector3f cam_pos_3f;
Vector3f cam_dir_3f;

////////////////////////////////////////////////////////////////////////////////

/* The following are parameters for creating an interactive first-person camera
 * view of the scene. The variables will make more sense when explained in
 * context, so you should just look at the 'mousePressed', 'mouseMoved', and
 * 'keyPressed' functions for the details.
 */

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

const float step_size = 0.2; // 0.005
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

static GLenum texture_image, texture_normals;
static GLint texture_image_pos, texture_normals_pos, tangent_attribute;

string shader_mode;

bool generate_normals = false;
bool create_texture_coords_buffer = false;

bool use_textures = true;

float h;
int smooth_toggle = false;
bool smooth_turned_off = false;

int q_rotation_setting = 0;

////////////////////////////////////////////////////////////////////////////////

/* Initialze OpenGL system and organize scene data (camera, lights, and
 * objects).
 */
void init(string scene_filename, string texture_image_filename,
          string texture_normals_filename, string shader_mode) {
    if (scene_filename == "-1") {
        create_lights();
        create_camera();
        cam.pos[0] = 0;
        cam.pos[1] = 0;
        cam.pos[2] = 3;
        create_square();
        original_objects = objects;
    }
    else {
        lights = read_lights(scene_filename);
        cam = read_camera(scene_filename);
        cam.pos[0] = 0;
        cam.pos[1] = 0;
        cam.pos[2] = 3;
        objects = read_objects(scene_filename, generate_normals,
                               create_texture_coords_buffer);
        original_objects = read_objects(scene_filename, generate_normals,
                                        create_texture_coords_buffer);
    }

    cam.fov = 60.0;
    cam.aspect = (float) xres / yres;

    cerr << "Loading textures" << endl;
    if(!(texture_image = readpng(texture_image_filename.c_str())))
       exit(1);
    if(!(texture_normals = readpng(texture_normals_filename.c_str())))
        exit(1);

    if (shader_mode == "0") {
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
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    glFrustum(cam.l, cam.r, cam.b, cam.t, cam.n, cam.f);

    glMatrixMode(GL_MODELVIEW);

    init_lights();

    last_rotation = quaternion(1, 0, 0, 0);
    current_rotation = quaternion(1, 0, 0, 0);

    transform_objects(true);
    glClearColor(101.0/255, 218.0/255, 255.0/255, 1);
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

   // Pass the total number of lights in the scene into the shader program.
   n_lights = glGetUniformLocation(shaderProgram, "n_lights");
   glUniform1i(n_lights, (int) lights.size());

   // Pass the texture image into the shader program.
   texture_image_pos = glGetUniformLocation(shaderProgram, "texture_image");
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture_image);
   glUniform1i(texture_image_pos, 0);

   // Pass the texture normal vectors file into the shader program.
   texture_normals_pos = glGetUniformLocation(shaderProgram, "texture_normals");
   glActiveTexture(GL_TEXTURE1);
   glBindTexture(GL_TEXTURE_2D, texture_normals);
   glUniform1i(texture_normals_pos, 1);

   // Set up the tangent vector attribute for the shader program.
   tangent_attribute = glGetAttribLocation(shaderProgram, "tangent");
}

/**
 * Initialze a light for the lights vector for the scene.
 */
void create_lights() {
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
void create_camera() {
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
void create_square() {
    Object square;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Reflectances
    ///////////////////////////////////////////////////////////////////////////////////////////////

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

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Points
    ///////////////////////////////////////////////////////////////////////////////////////////////

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

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Normals
    ///////////////////////////////////////////////////////////////////////////////////////////////

    Vec3f normal1;
    normal1.x = 0;
    normal1.y = 0;
    normal1.z = 1;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Vertex and Normal Arrays
    ///////////////////////////////////////////////////////////////////////////////////////////////

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
    objects.push_back(square);
}

/**
 * Reshape the image whenever the window size changes.
 */
void reshape(int width, int height) {
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
void display(void) {
    // cout << "HERE 1" << endl;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(101.0/255, 218.0/255, 255.0/255, 1);

    glLoadIdentity();

    float ori_mag = sqrt(cam.ori_axis[0] * cam.ori_axis[0]
                         + cam.ori_axis[1] * cam.ori_axis[1]
                         + cam.ori_axis[2] * cam.ori_axis[2]);
    // glRotatef(-cam.ori_angle * 180 / M_PI,
    //           cam.ori_axis[0] / ori_mag, cam.ori_axis[1] / ori_mag,
    //           cam.ori_axis[2] / ori_mag);
    glRotatef(cam.ori_angle, // camera angle in given in degrees
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

    // glMultMatrixf(rotation_matrix);

    Matrix4f q_v_transform;
    // q_v_transform <<
    //      rotation_matrix[0], rotation_matrix[1],
    //      rotation_matrix[2], rotation_matrix[3],
    //
    //      rotation_matrix[4], rotation_matrix[5],
    //      rotation_matrix[6], rotation_matrix[7],
    //
    //      rotation_matrix[8], rotation_matrix[9],
    //      rotation_matrix[10], rotation_matrix[11],
    //
    //      rotation_matrix[12], rotation_matrix[13],
    //      rotation_matrix[14], rotation_matrix[15];
    q_v_transform <<
         rotation_matrix[0], rotation_matrix[4],
         rotation_matrix[8], rotation_matrix[12],

         rotation_matrix[1], rotation_matrix[5],
         rotation_matrix[9], rotation_matrix[13],

         rotation_matrix[2], rotation_matrix[6],
         rotation_matrix[10], rotation_matrix[14],

         rotation_matrix[3], rotation_matrix[7],
         rotation_matrix[11], rotation_matrix[15];

    Matrix4f q_v_transform_inv = q_v_transform.inverse();
    Matrix4f q_n_transform = q_v_transform_inv.transpose();

    Vector4f cam_pos_4f;
    cam_pos_4f << cam_pos_3f(0), cam_pos_3f(0), cam_pos_3f(0), 1.0;
    cam_pos_4f = q_v_transform * cam_pos_4f;
    cam_pos_3f(0) = cam_pos_4f(0) / cam_pos_4f(3);
    cam_pos_3f(1) = cam_pos_4f(1) / cam_pos_4f(3);
    cam_pos_3f(2) = cam_pos_4f(2) / cam_pos_4f(3);

    Vector4f cam_dir_4f;
    cam_dir_4f << cam_dir_3f(0), cam_dir_3f(0), cam_dir_3f(0), 1.0;
    cam_dir_4f = q_v_transform * cam_dir_4f;
    cam_dir_3f(0) = cam_dir_4f(0) / cam_dir_4f(3);
    cam_dir_3f(1) = cam_dir_4f(1) / cam_dir_4f(3);
    cam_dir_3f(2) = cam_dir_4f(2) / cam_dir_4f(3);

    // cout << "HERE 2" << endl;

    if (q_rotation_setting != 2) {
    for (int i = 0; i < (int) objects.size(); i++) {
        // Transform all the vertices in the given vector of vertices.
        vector<Vertex> v_vector = objects[i].world_space_vertices;

        vector<Vertex> t_vertices;
        vector<Vec3f> t_normals;

        for (int k = 0; k < (int) v_vector.size(); k++) {
            Vector4f v;
            v << v_vector[k].x, v_vector[k].y, v_vector[k].z, 1.0;
            Vector4f tv;
            tv = q_v_transform * v;
            Vertex t_vertex;
            t_vertex.x = tv(0) / tv(3);
            t_vertex.y = tv(1) / tv(3);
            t_vertex.z = tv(2) / tv(3);
            t_vertices.push_back(t_vertex);
        }

        vector<Vec3f> vn_vector = objects[i].world_space_normals;
        for (int k = 0; k < (int) vn_vector.size(); k++) {
            Vector4f vn;
            vn << vn_vector[k].x, vn_vector[k].y, vn_vector[k].z, 1.0;
            Vector4f tvn;
            tvn = q_n_transform * vn;
            Vec3f t_vnorm;
            t_vnorm.x = tvn(0) / tvn(3);
            t_vnorm.y = tvn(1) / tvn(3);
            t_vnorm.z = tvn(2) / tvn(3);
            t_normals.push_back(t_vnorm);
        }

        objects[i].transformed_vertices = t_vertices;
        objects[i].transformed_normals = t_normals;
    }
    }

    if (q_rotation_setting != 1) {
    for (int i = 0; i < (int) lights.size(); i++) {
        Vector4f l_pos;
        l_pos << lights[i].fixed_pos[0], lights[i].fixed_pos[1], lights[i].fixed_pos[2],
                 lights[i].fixed_pos[3];
        Vector4f t_l_pos;
        t_l_pos = q_v_transform * l_pos;
        lights[i].pos[0] = t_l_pos(0);
        lights[i].pos[1] = t_l_pos(1);
        lights[i].pos[2] = t_l_pos(2);
        lights[i].pos[3] = t_l_pos(3);
    }
    }

    // cout << "HERE 3" << endl;

    set_lights();

    // glEnable(GL_CULL_FACE);
    // glCullFace(GL_FRONT);

    // draw_object_outlines();

    // glEnable(GL_CULL_FACE);
    // glCullFace(GL_BACK);

    // cout << "HERE 4" << endl;

    draw_objects();

    glutSwapBuffers();

    // cout << "HERE 5" << endl;
}

/**
 * Initialze the lights of the scene.
 */
void init_lights() {
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
void set_lights() {
    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {
        int light_id = GL_LIGHT0 + i;

        glLightfv(light_id, GL_POSITION, lights[i].pos);
    }
}

/**
 * Transform the objects into world space in advance, so the ray tracing
 * function can have access to the world space coordinates of the objects.
 * Additional note: maybe quaternion rotation should be applied here after
 * object transformations are applied here.
 */
void transform_objects(bool init) {
    int num_objects = objects.size();
    for (int i = 0; i < num_objects; i++) {
        if (init) {
            int num_transform_sets = objects[i].transform_sets.size();
            for (int j = num_transform_sets - 1; j >= 0 ; j--) {
                Obj_Transform transform = objects[i].transform_sets[j];
                Matrix4f new_matrix;
                if (transform.type == "t") {
                    new_matrix << 1, 0, 0, transform.components[0],
                                  0, 1, 0, transform.components[1],
                                  0, 0, 1, transform.components[2],
                                  0, 0, 0, 1;
                    objects[i].t_transform_matrices.push_back(new_matrix);
                }
                else if (transform.type == "r") {
                    float rx = transform.components[0];
                    float ry = transform.components[1];
                    float rz = transform.components[2];
                    float angle = transform.rotation_angle;
                    float rotation_axis_magnitude = rx * rx + ry * ry + rz * rz;
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
                    new_matrix << a00, a01, a02, 0,
                                  a10, a11, a12, 0,
                                  a20, a21, a22, 0,
                                  0, 0, 0, 1;
                    objects[i].t_transform_matrices.push_back(new_matrix);
                    objects[i].n_transform_matrices.push_back(new_matrix);
                }
                else {
                    glScalef(transform.components[0], transform.components[1],
                             transform.components[2]);
                    new_matrix << transform.components[0], 0, 0, 0,
                                  0, transform.components[1], 0, 0,
                                  0, 0, transform.components[2], 0,
                                  0, 0, 0, 1;
                    objects[i].t_transform_matrices.push_back(new_matrix);
                    objects[i].n_transform_matrices.push_back(new_matrix);
                }
            }
        }

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
        vector<Matrix4f> tm_vector = objects[i].t_transform_matrices;
        for (int i = 0; i < (int) tm_vector.size(); i++) {
            v_transform = tm_vector[i] * v_transform;
        }

        vector<Matrix4f> nm_vector = objects[i].n_transform_matrices;
        for (int i = 0; i < (int) nm_vector.size(); i++) {
            n_transform = nm_vector[i] * n_transform;
        }

        Matrix4f n_transform_inv = n_transform.inverse();
        Matrix4f vn_transform = n_transform_inv.transpose();

        // Transform all the vertices in the given vector of vertices.
        vector<Vertex> v_vector = objects[i].vertex_buffer;

        vector<Vertex> t_vertices;
        vector<Vec3f> t_normals;

        for (int k = 0; k < (int) v_vector.size(); k++) {
            Vector4f v;
            v << v_vector[k].x, v_vector[k].y, v_vector[k].z, 1.0;
            Vector4f tv;
            tv = v_transform * v;
            Vertex t_vertex;
            t_vertex.x = tv(0) / tv(3);
            t_vertex.y = tv(1) / tv(3);
            t_vertex.z = tv(2) / tv(3);
            t_vertices.push_back(t_vertex);
        }

        vector<Vec3f> vn_vector = objects[i].normal_buffer;
        for (int k = 0; k < (int) vn_vector.size(); k++) {
            Vector4f vn;
            vn << vn_vector[k].x, vn_vector[k].y, vn_vector[k].z, 1.0;
            Vector4f tvn;
            tvn = vn_transform * vn;
            Vec3f t_vnorm;
            t_vnorm.x = tvn(0) / tvn(3);
            t_vnorm.y = tvn(1) / tvn(3);
            t_vnorm.z = tvn(2) / tvn(3);
            t_normals.push_back(t_vnorm);
        }

        objects[i].world_space_vertices = t_vertices;
        objects[i].world_space_normals = t_normals;
        objects[i].transformed_vertices = t_vertices;
        objects[i].transformed_normals = t_normals;
    }
}

/**
 * Draw the objects in the scene. New note: add functionality to add array of
 * all triangle faces of all objects for ray tracing function to find ray-
 * object intersections.
 */
void draw_object_outlines() {
    int num_objects = objects.size();

    for(int i = 0; i < num_objects; ++i)
    {
        // glPushMatrix();

        {
            // int num_transform_sets = objects[i].transform_sets.size();

            /* Modify the current modelview matrix with the
             * geometric transformations for this object.
             */
            // for(int j = num_transform_sets - 1; j >= 0 ; j--) {
            //     Obj_Transform transform = objects[i].transform_sets[j];
            //     if (transform.type == "t") {
            //         glTranslatef(transform.components[0],
            //                      transform.components[1],
            //                      transform.components[2]);
            //     }
            //     else if (transform.type == "r") {
            //         glRotatef(transform.rotation_angle,
            //                   transform.components[0], transform.components[1],
            //                   transform.components[2]);
            //     }
            //     else {
            //         glScalef(transform.components[0], transform.components[1],
            //                  transform.components[2]);
            //     }
            // }

            // float outline_color[3]; // red
            // outline_color[0] = 1;
            // outline_color[1] = 0;
            // outline_color[2] = 0;
            // float outline_shininess = 0;
            // float outline_color2[3]; // red
            // outline_color2[0] = 0;
            // outline_color2[1] = 1;
            // outline_color2[2] = 0;
            // float outline_shininess2 = 1;
            // glMaterialfv(GL_FRONT, GL_AMBIENT, outline_color);
            // glMaterialfv(GL_FRONT, GL_DIFFUSE, outline_color2);
            // glMaterialfv(GL_FRONT, GL_SPECULAR, outline_color2);
            // glMaterialf(GL_FRONT, GL_SHININESS, outline_shininess);

            float outline_color[3]; // black
            outline_color[0] = 0;
            outline_color[1] = 0;
            outline_color[2] = 0;
            float outline_shininess = 0;
            glMaterialfv(GL_FRONT, GL_AMBIENT, outline_color);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, outline_color);
            glMaterialfv(GL_FRONT, GL_SPECULAR, outline_color);
            glMaterialf(GL_FRONT, GL_SHININESS, outline_shininess);

            Matrix4f outline_matrix;
            outline_matrix << 1.1, 0, 0, 0,
                              0, 1.1, 0, 0,
                              0, 0, 1.1, 0,
                              0, 0, 0, 1;

            vector<Vertex> outline_vertices;
            for (int j = 0; j < (int) objects[i].transformed_vertices.size();
                 j+=3) {
                // Vector4f outline_vertex4f1;
                // outline_vertex4f1 << objects[i].transformed_vertices[j].x,
                //                      objects[i].transformed_vertices[j].y,
                //                      objects[i].transformed_vertices[j].z, 1;
                // outline_vertex4f1 = outline_matrix * outline_vertex4f1;
                // Vertex outline_vertex1;
                // outline_vertex1.x = outline_vertex4f1(0) / outline_vertex4f1(3);
                // outline_vertex1.y = outline_vertex4f1(1) / outline_vertex4f1(3);
                // outline_vertex1.z = outline_vertex4f1(2) / outline_vertex4f1(3);
                //
                // Vector4f outline_vertex4f2;
                // outline_vertex4f2 << objects[i].transformed_vertices[j+1].x,
                //                      objects[i].transformed_vertices[j+1].y,
                //                      objects[i].transformed_vertices[j+1].z, 1;
                // outline_vertex4f2 = outline_matrix * outline_vertex4f2;
                // Vertex outline_vertex2;
                // outline_vertex2.x = outline_vertex4f2(0) / outline_vertex4f2(3);
                // outline_vertex2.y = outline_vertex4f2(1) / outline_vertex4f2(3);
                // outline_vertex2.z = outline_vertex4f2(2) / outline_vertex4f2(3);
                //
                // Vector4f outline_vertex4f3;
                // outline_vertex4f3 << objects[i].transformed_vertices[j+2].x,
                //                   objects[i].transformed_vertices[j+2].y,
                //                   objects[i].transformed_vertices[j+2].z, 1;
                // outline_vertex4f3 = outline_matrix * outline_vertex4f3;
                // Vertex outline_vertex3;
                // outline_vertex3.x = outline_vertex4f3(0) / outline_vertex4f3(3);
                // outline_vertex3.y = outline_vertex4f3(1) / outline_vertex4f3(3);
                // outline_vertex3.z = outline_vertex4f3(2) / outline_vertex4f3(3);

                float x_mult = 0.01;
                float y_mult = 0.01;
                float z_mult = 0.01;

                Vec3f vnorm1 = objects[i].transformed_normals[j];
                Vertex outline_vertex1 = objects[i].transformed_vertices[j];
                outline_vertex1.x += vnorm1.x * x_mult;
                outline_vertex1.y += vnorm1.y * y_mult;
                outline_vertex1.z += vnorm1.z * z_mult;

                Vec3f vnorm2 = objects[i].transformed_normals[j+1];
                Vertex outline_vertex2 = objects[i].transformed_vertices[j+1];
                outline_vertex2.x += vnorm2.x * x_mult;
                outline_vertex2.y += vnorm2.y * y_mult;
                outline_vertex2.z += vnorm2.z * z_mult;

                Vec3f vnorm3 = objects[i].transformed_normals[j+2];
                Vertex outline_vertex3 = objects[i].transformed_vertices[j+2];
                outline_vertex3.x += vnorm3.x * x_mult;
                outline_vertex3.y += vnorm3.y * y_mult;
                outline_vertex3.z += vnorm3.z * z_mult;

                outline_vertices.push_back(outline_vertex2);
                outline_vertices.push_back(outline_vertex1);
                outline_vertices.push_back(outline_vertex3);
            }

            // vector<Vec3f> inverted_normals;
            // for (int j = 0; j < (int) objects[i].transformed_vertices.size();
            //      j++) {
            //
            //     Vec3f normal = objects[i].transformed_normals[j];
            //     Vec3f inverted_normal;
            //     inverted_normal.x = normal.x;
            //     inverted_normal.y = normal.y;
            //     inverted_normal.z = normal.z;
            //     inverted_normals.push_back(inverted_normal);
            // }

            // glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);
            glVertexPointer(3, GL_FLOAT, 0, &outline_vertices[0]);

            // glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);
            // glNormalPointer(GL_FLOAT, 0, &inverted_normals[0]);
            glNormalPointer(GL_FLOAT, 0, &objects[i].transformed_normals[0]);

            int buffer_size = objects[i].vertex_buffer.size();

            // if(!wireframe_mode)
                glDrawArrays(GL_TRIANGLES, 0, buffer_size);
            // else
            //     for(int j = 0; j < buffer_size; j += 3)
            //         glDrawArrays(GL_LINE_LOOP, j, 3);
        }

        // glPopMatrix();
    }
}

// /**
//  * Draw the objects in the scene. New note: add functionality to add array of
//  * all triangle faces of all objects for ray tracing function to find ray-
//  * object intersections.
//  */
// void draw_objects()
// {
//     int num_objects = objects.size();
//
//     for(int i = 0; i < num_objects; ++i)
//     {
//         glPushMatrix();
//
//         {
//             int num_transform_sets = objects[i].transform_sets.size();
//
//             /* Modify the current modelview matrix with the
//              * geometric transformations for this object.
//              */
//             for(int j = num_transform_sets - 1; j >= 0 ; j--) {
//                 Obj_Transform transform = objects[i].transform_sets[j];
//                 if (transform.type == "t") {
//                     glTranslatef(transform.components[0],
//                                  transform.components[1],
//                                  transform.components[2]);
//                 }
//                 else if (transform.type == "r") {
//                     glRotatef(transform.rotation_angle,
//                               transform.components[0], transform.components[1],
//                               transform.components[2]);
//                 }
//                 else {
//                     glScalef(transform.components[0], transform.components[1],
//                              transform.components[2]);
//                 }
//             }
//
//             glMaterialfv(GL_FRONT, GL_AMBIENT, objects[i].ambient_reflect);
//             glMaterialfv(GL_FRONT, GL_DIFFUSE, objects[i].diffuse_reflect);
//             glMaterialfv(GL_FRONT, GL_SPECULAR, objects[i].specular_reflect);
//             glMaterialf(GL_FRONT, GL_SHININESS, objects[i].shininess);
//
//             glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);
//             // glVertexPointer(3, GL_FLOAT, 0, &objects[i].transformed_vertices[0]);
//
//             glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);
//             // glNormalPointer(GL_FLOAT, 0, &objects[i].transformed_normals[0]);
//
//             /**
//              * Create the array of texture coordinates to be used in the shader
//              * program. THIS NEEDS TO BE MODIFIED!!!!!
//              */
//             GLfloat texture_buffer[] = {
//                 0, 0,
//                 1, 0,
//                 1, 1,
//                 0, 0,
//                 1, 1,
//                 0, 1,
//             };
//
//             /* Pointer for the texture coordinates to be read in the shader
//              * program.
//              */
//             glTexCoordPointer(2, GL_FLOAT, 0, texture_buffer);
//
//             /**
//              * Create the array of tangent vectors to be used in the shader
//              * program. THIS NEEDS TO BE MODIFIED!!!!!
//              */
//             GLfloat tangent_buffer[] = {
//                 -1, 0, 0,
//                 -1, 0, 0,
//                 -1, 0, 0,
//                 -1, 0, 0,
//             };
//
//             /* Pointer for the tangent vectors to be read in the shader
//              * program.
//              */
//             glVertexAttribPointer(tangent_attribute, 3, GL_FLOAT, GL_FALSE,
//                                   0, 0);
//
//             int buffer_size = objects[i].vertex_buffer.size();
//
//             if(!wireframe_mode)
//                 glDrawArrays(GL_TRIANGLES, 0, buffer_size);
//             else
//                 for(int j = 0; j < buffer_size; j += 3)
//                     glDrawArrays(GL_LINE_LOOP, j, 3);
//         }
//
//         glPopMatrix();
//     }
// }

/**
 * Draw the objects in the scene. New note: add functionality to add array of
 * all triangle faces of all objects for ray tracing function to find ray-
 * object intersections.
 */
void draw_objects() {
    int num_objects = objects.size();

    for(int i = 0; i < num_objects; ++i)
    {
        // glPushMatrix();

        {
            // int num_transform_sets = objects[i].transform_sets.size();

            /* Modify the current modelview matrix with the
             * geometric transformations for this object.
             */
            // for(int j = num_transform_sets - 1; j >= 0 ; j--) {
            //     Obj_Transform transform = objects[i].transform_sets[j];
            //     if (transform.type == "t") {
            //         glTranslatef(transform.components[0],
            //                      transform.components[1],
            //                      transform.components[2]);
            //     }
            //     else if (transform.type == "r") {
            //         glRotatef(transform.rotation_angle,
            //                   transform.components[0], transform.components[1],
            //                   transform.components[2]);
            //     }
            //     else {
            //         glScalef(transform.components[0], transform.components[1],
            //                  transform.components[2]);
            //     }
            // }

            glMaterialfv(GL_FRONT, GL_AMBIENT, objects[i].ambient_reflect);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, objects[i].diffuse_reflect);
            glMaterialfv(GL_FRONT, GL_SPECULAR, objects[i].specular_reflect);
            glMaterialf(GL_FRONT, GL_SHININESS, objects[i].shininess);

            // glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);
            glVertexPointer(3, GL_FLOAT, 0, &objects[i].transformed_vertices[0]);

            // glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);
            glNormalPointer(GL_FLOAT, 0, &objects[i].transformed_normals[0]);

            GLfloat texture_buffer[] = {
                0, 0,
                1, 0,
                1, 1,
                0, 0,
                1, 1,
                0, 1,
            };
            // GLfloat tangent_buffer[] = {
            //     -1, 0, 0,
            //     -1, 0, 0,
            //     -1, 0, 0,
            //     -1, 0, 0,
            // };

            if (use_textures) {
            /**
             * Create the array of texture coordinates to be used in the shader
             * program. THIS NEEDS TO BE MODIFIED!!!!!
             */

            // int count_tangent_vectors = 4;
            // while (count_tangent_vectors < (int) objects[0].vertex_buffer.size()) {
            //     tangent_buffer
            // }

            // cout << "HERE 6" << endl;

            /* Pointer for the texture coordinates to be read in the shader
             * program.
             */
            glTexCoordPointer(2, GL_FLOAT, 0, texture_buffer);
            // glTexCoordPointer(2, GL_FLOAT, 0,
            //                   &objects[i].texture_coords_buffer[0]);

            /**
             * Create the array of tangent vectors to be used in the shader
             * program. THIS NEEDS TO BE MODIFIED!!!!!
             */


            /* Pointer for the tangent vectors to be read in the shader
             * program.
             */
            glVertexAttribPointer(tangent_attribute, 3, GL_FLOAT, GL_FALSE,
                                  0, 0);
            }

            int buffer_size = objects[i].vertex_buffer.size();

            // cout << "HERE 7" << endl;

            if(!wireframe_mode)
                glDrawArrays(GL_TRIANGLES, 0, buffer_size);
            else
                for(int j = 0; j < buffer_size; j += 3)
                    glDrawArrays(GL_LINE_LOOP, j, 3);
        }

        // cout << "HERE 8" << endl;

        // glPopMatrix();
    }
}

/**
 * Function to tell OpenGL what to do when the (left) mouse (button) is pressed.
 */
void mouse_pressed(int button, int state, int x, int y) {
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
void mouse_moved(int x, int y) {
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
float deg2rad(float angle) {
    return angle * M_PI / 180.0;
}

/**
 * Function tells OpenGL what to do when a key is pressed.
 */
void key_pressed(unsigned char key, int x, int y) {

    if(key == 'q') {
        exit(0);
    }

    else if(key == 't') {
        wireframe_mode = !wireframe_mode;

        glutPostRedisplay();
    }
    /* If the 'h' key is pressed, toggle smoothing on or off depending on the
     * previous state of smooth_toggle.
     */
    else if(key == 'h') {
        smooth_toggle = !smooth_toggle;
        if (!smooth_toggle) {
            smooth_turned_off = true;
        }
    }
    /* If the 'u' key is pressed, increment the time step h by 0.0001 and revert
     * back to the original object.
     */
    else if (key == 'u') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h += 0.0001;
        cout << "h incremented by 0.0001: h = " << h << endl;
    }
    /* If the 'i' key is pressed, decrement the time step h by 0.0001 and revert
     * back to the original object.
     */
    else if (key == 'i') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h -= 0.0001;
        cout << "h decremented by 0.0001: h = " << h << endl;
    }
    /* If the 'o' key is pressed, double the time step h and revert back to the
     * original object.
     */
    else if (key == 'o') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h *= 2;
        cout << "h doubled: h = " << h << endl;
    }
    /* If the 'p' key is pressed, halve the time step h and revert back to the
     * original object.
     */
    else if (key == 'p') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h /= 2;
        cout << "h halved: h = " << h << endl;
    }
    /* If the 'j' key is pressed, set the time step h to 0 and revert back to
     * the original object.
     */
    else if (key == 'j') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h = 0;
        cout << "set h = " << h << endl;
    }
    // Key for ray tracing while using OpenGL.
    // else if (key == 'r') {
    //     // Ray camera_ray;
    //     // camera_ray.origin_x = 0;
    //     // camera_ray.origin_y = 0;
    //     // camera_ray.origin_z = 3;
    //     // camera_ray.direction_x = 0;
    //     // camera_ray.direction_y = 0;
    //     // camera_ray.direction_z = -1;
    //     // vector<Intersection> intersections;
    //     // findAllIntersections(intersections, camera_ray);
    //     // raytrace();
    // }
    else if (key == 'l') {
        if (q_rotation_setting == 0) {
            cout << "Rotating objects only." << endl;
            q_rotation_setting = 1;
        }
        else if (q_rotation_setting == 1) {
            cout << "Rotating lights only." << endl;
            q_rotation_setting = 2;
        }
        else {
            cout << "Rotating both objects and lights." << endl;
            q_rotation_setting = 0;
        }
    }
    else {
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

    if (key == 'h' || key == 'u' || key == 'i' || key == 'o' || key == 'p'
        || key == 'j') {
        /* If smooth_toggle == false, revert back to the original object for each
         * object in the scene. Otherwise, smooth all the objects in the scene
         * using implicit fairing with the current time step h.
         */
        if (!smooth_toggle) {
            if (smooth_turned_off) {
                cout << "No smoothing." << endl;
                smooth_turned_off = false;
            }

            // Revert to original objects.
            objects = original_objects;
        }
        else {
            cout << "Smoothing with h = " << h << "." << endl;
            for (int i = 0; i < (int) objects.size(); i++) {
                smooth_object(objects[i], h);
            }
            cout << "Smoothing done!" << endl;
        }
        display();
    }
}

/* The 'main' function:
 *
 * Run the OpenGL program (initialize OpenGL, display the scene, and react to
 * mouse and keyboard presses).
 */
int main(int argc, char* argv[])
{
    if (argc != 7) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [scene_description_file.txt]"
                                 + " [color_texture.png] [normal_map.png]"
                                 + " [xres] [yres] [shader_mode]";
        cout << usage_statement << endl;
        return 1;
    }

    string scene_filename = argv[1];
    string texture_image_filename = argv[2];
    string texture_normals_filename = argv[3];

    xres = atoi(argv[4]); // 800
    yres = atoi(argv[5]); // 800

    // mode = 0 for Gouraud shading, mode = 1 for Phong shading.
    shader_mode = argv[6];

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(xres, yres);

    glutInitWindowPosition(0, 0);

    glutCreateWindow("Illustrative Rendering");

    if (shader_mode != "0") {
        vertProgFileName = "shaders/vertexProgram_" + shader_mode + ".glsl";
        fragProgFileName = "shaders/fragmentProgram_" + shader_mode + ".glsl";
    }
    init(scene_filename, texture_image_filename, texture_normals_filename,
         shader_mode);

    glutDisplayFunc(display);

    glutReshapeFunc(reshape);

    glutMouseFunc(mouse_pressed);

    glutMotionFunc(mouse_moved);

    glutKeyboardFunc(key_pressed);

    glutMainLoop();
}
