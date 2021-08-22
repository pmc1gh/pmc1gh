// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
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

void init(string scene_filename);
extern GLenum readpng(const char *filename);
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

////////////////// Global Data Structures and Constants ////////////////////////

// Camera and lists of lights and objects.

Camera cam;
vector<Point_Light> lights;
vector<Object> objects;
vector<Object> original_objects;

Vector3f cam_pos_3f;
Vector3f cam_dir_3f;

/* The following are parameters for creating an interactive first-person camera
 * view of the scene. The variables will make more sense when explained in
 * context, so you should just look at the 'mousePressed', 'mouseMoved', and
 * 'keyPressed' functions for the details.
 */

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

const float step_size = 0.01;
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

// static GLenum texture_image, texture_normals;
vector<GLenum> texture_image_vector;
vector<GLenum> texture_normals_vector;
static GLint texture_image_pos, texture_normals_pos,
             warped_diffuse_texture_image_pos, tangent_attribute;

string warped_diffuse_texture_file;
GLenum warped_diffuse_texture_image;

// Shader mode determines which set of shaders are used.
string shader_mode;

bool generate_normals = true;
bool create_texture_coords_buffer = true;

bool use_textures = true;

float h;
int smooth_toggle = false;
bool smooth_turned_off = false;

int rotation_setting = 0;

string scene_filename;

float rotation_matrix[16];
float last_rotation_matrix[16];

// Animation-related global variables
float global_time;
float target_fps = 30;
float time_per_frame = 1000 / target_fps;
float current_frame_number;
float animate_rotation = 0;
float current_animation_time = 0;
float rotation_speed_scale = 0.1;

// height_map Class
height_map hm;

// height_map animation function
function<float(float, float)> wave_function = [](float x, float y)->float{
    return (1.0 / 10.0)
           * (sin(4 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5)
                  - current_animation_time / 50)))
           + (1.5 / 10.0)
           * sin(1 * ((x + 0.5) * (x + 0.5) + (y + 0.5) * (y + 0.5)
                 - current_animation_time / 50));
};

////////////////////////////////////////////////////////////////////////////////

/* Initialze OpenGL system and organize scene data (camera, lights, and
 * objects).
 */
void init(string scene_filename) {
    global_time = 0;

    // Hardcoded debug mode object (textured square)
    if (scene_filename == "-1") {
        create_default_lights(lights);
        create_default_camera(cam);
        create_default_square(objects);
        original_objects = objects;
    }
    // Heightmap for fluid surface animation.
    else if (scene_filename == "fsa") {
        create_default_lights(lights);
        create_default_camera(cam);

        hm = height_map();
        objects.push_back(hm.get_hm_obj());
    }
    // Importing a set of 3D mesh models.
    else {
        lights = read_lights(scene_filename);
        cam = read_camera(scene_filename);
        objects = read_objects(scene_filename, generate_normals,
                               create_texture_coords_buffer);
        original_objects = objects;
        vector<float> obj0_bb = get_object_bounding_box(objects[0]);
        cout << "Object bounding box coordinates:\n"
             << obj0_bb[0] << " " << obj0_bb[1] << " " << obj0_bb[2] << " "
             << obj0_bb[3] << " " << obj0_bb[4] << " " << obj0_bb[5] << endl;
    }

    // Set the Camera FOV and aspect ratio parameters.
    cam.fov = 60.0;
    cam.aspect = (float) xres / yres;

    // Import texture images if applicable (when using texturing shaders)
    if (shader_mode != "sc" && shader_mode != "sc_glsl") {

        cerr << "Loading textures" << endl;

        // Creating vectors to store all texture images and normal maps.
        GLenum texture_image, texture_normals;
        for (int i = 0; i < (int) objects.size(); i++) {
            if(!(texture_image = readpng(objects[i].texture_image.c_str())))
                exit(1);
            if(!(texture_normals = readpng(objects[i].texture_normals.c_str())))
                exit(1);
            texture_image_vector.push_back(texture_image);
            texture_normals_vector.push_back(texture_normals);
        }

        /* Importing warped diffuse texture for diffuse lighting as described in
         * the Team Fortess 2 Artstyle paper.
         */
        warped_diffuse_texture_file =
            "../data/textures/warped_diffuse_texture2d_stretched.png";
        warped_diffuse_texture_image =
            readpng(warped_diffuse_texture_file.c_str());
    }

    /* Use OpenGL's smooth shading (Gouraud shading) when applicable. Otherwise,
     * read in and compile the applicable shader programs.
     */
    if (shader_mode == "sc") {
        glShadeModel(GL_SMOOTH);
    }
    else {
        readShaders();
    }

    // Have OpenGL not render hidden faces.
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_DEPTH_TEST);

    // More OpenGL setup...
    glEnable(GL_NORMALIZE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    if (shader_mode != "sc" && shader_mode != "sc_glsl") {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    }

    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    glFrustum(cam.l, cam.r, cam.b, cam.t, cam.n, cam.f);

    glMatrixMode(GL_MODELVIEW);

    // Initialize and set lights in scene.
    init_lights();
    set_lights();

    /* Initialize rotation quaternions for arcball object/light rotaions in
     * scene.
     */
    last_rotation = quaternion(1, 0, 0, 0);
    current_rotation = quaternion(1, 0, 0, 0);

    // Set color to cyan/sky blue for scenes using texturing shaders.
    if (shader_mode != "sc" && shader_mode != "sc_glsl") {
        glClearColor(101.0/255, 218.0/255, 255.0/255, 1);
    }

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
   glUniform1i(texture_image_pos, 0);

   // Pass the texture normal vectors file into the shader program.
   texture_normals_pos = glGetUniformLocation(shaderProgram, "texture_normals");
   glUniform1i(texture_normals_pos, 1);

   // Pass the warped diffuse texture image into the shader program.
   warped_diffuse_texture_image_pos = glGetUniformLocation(shaderProgram,
                                            "warped_diffuse_texture");
   glUniform1i(warped_diffuse_texture_image_pos, 2);

   // Activate the warped diffuse lighting texture image here by default.
   glActiveTexture(GL_TEXTURE2);
   glBindTexture(GL_TEXTURE_2D, warped_diffuse_texture_image);

   // Set up the tangent vector attribute for the shader program.
   tangent_attribute = glGetAttribLocation(shaderProgram, "tangent");
   glEnableVertexAttribArray(tangent_attribute);
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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (shader_mode != "sc" && shader_mode != "sc_glsl") {
        glClearColor(101.0/255, 218.0/255, 255.0/255, 1);
    }

    glLoadIdentity();

    float ori_mag = sqrt(cam.ori_axis[0] * cam.ori_axis[0]
                         + cam.ori_axis[1] * cam.ori_axis[1]
                         + cam.ori_axis[2] * cam.ori_axis[2]);
    glRotatef(cam.ori_angle, // camera angle in given in degrees
              cam.ori_axis[0] / ori_mag, cam.ori_axis[1] / ori_mag,
              cam.ori_axis[2] / ori_mag);

    glTranslatef(-cam.pos[0], -cam.pos[1], -cam.pos[2]);

    if (rotation_setting == 2) {
        glMultMatrixf(last_rotation_matrix);
        draw_objects();
    }

    quaternion rotation = current_rotation.q_mult(last_rotation);
    float qs = rotation.get_s();
    float *v = rotation.get_v();
    float qx = v[0];
    float qy = v[1];
    float qz = v[2];
    rotation_matrix[0] = 1 - 2 * qy * qy - 2 * qz * qz;
    rotation_matrix[1] = 2 * (qx * qy + qz * qs);
    rotation_matrix[2] = 2 * (qx * qz - qy * qs);
    rotation_matrix[3] = 0;
    rotation_matrix[4] = 2 * (qx * qy - qz * qs);
    rotation_matrix[5] = 1 - 2 * qx * qx - 2 * qz * qz;
    rotation_matrix[6] = 2 * (qy * qz + qx * qs);
    rotation_matrix[7] = 0;
    rotation_matrix[8] = 2 * (qx * qz + qy * qs);
    rotation_matrix[9] = 2 * (qy * qz - qx * qs);
    rotation_matrix[10] = 1 - 2 * qx * qx - 2 * qy * qy;
    rotation_matrix[11] = 0;
    rotation_matrix[12] = 0;
    rotation_matrix[13] = 0;
    rotation_matrix[14] = 0;
    rotation_matrix[15] = 1;

    glMultMatrixf(rotation_matrix);

    if (rotation_setting != 1) {
        set_lights();
    }

    if (rotation_setting != 2) {
        draw_objects();
    }

    glutSwapBuffers();
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
 * Draw the objects in the scene.
 */
void draw_objects() {
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
                Obj_Transform transform = objects[i].transform_sets[j];
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

            // Animation rotation
            // glRotatef(current_animation_time * rotation_speed_scale, 0, 1, 0);

            glMaterialfv(GL_FRONT, GL_AMBIENT, objects[i].ambient_reflect);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, objects[i].diffuse_reflect);
            glMaterialfv(GL_FRONT, GL_SPECULAR, objects[i].specular_reflect);
            glMaterialf(GL_FRONT, GL_SHININESS, objects[i].shininess);

            glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);

            glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);

            if (use_textures) {
                if (shader_mode != "sc" && shader_mode != "sc_glsl") {
                    // Set textures for current object.
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, texture_image_vector[i]);

                    glActiveTexture(GL_TEXTURE1);
                    glBindTexture(GL_TEXTURE_2D, texture_normals_vector[i]);

                    /* Pointer for the texture coordinates to be read in the
                     * shader program.
                     */
                    glTexCoordPointer(2, GL_FLOAT, 0,
                                      &objects[i].texture_coords_buffer[0]);


                    /* Pointer for the tangent vectors to be read in the shader
                     * program.
                     */
                    glVertexAttribPointer(tangent_attribute, 3, GL_FLOAT, GL_FALSE,
                                          0, &objects[i].tangent_buffer[0]);
                }
            }

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

    else if(key == 'r') {
        wireframe_mode = !wireframe_mode;

        glutPostRedisplay();
    }

    else if(key == 't') {
        cout << "Time elapsed: " << global_time << endl;
    }
    /* If the 'h' key is pressed, toggle smoothing on or off depending on the
     * previous state of smooth_toggle.
     *
     * Smoothing does not work with open meshes.
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
    else if (key == 'l') {
        if (rotation_setting == 0) {
            cout << "Rotating objects only." << endl;
            rotation_setting = 1;
        }
        else if (rotation_setting == 1) {
            cout << "Rotating lights only." << endl;
            rotation_setting = 2;
            for (int i = 0; i < 16; i++) {
                last_rotation_matrix[i] = rotation_matrix[i];
            }
        }
        else {
            cout << "Rotating both objects and lights." << endl;
            rotation_setting = 0;
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

    else if (key == 'z') {
        animate_rotation = ((int) animate_rotation + 1) % 2;
        if (animate_rotation) {
            cout << "Starting animation" << endl;
        }
        else {
            cout << "Stopping animation" << endl;
        }
    }
}

void main_loop() {
    global_time++;
    cout << "Time: " << global_time << endl;
}

void idle() {
    glutPostRedisplay();
    global_time++;
    if (animate_rotation) {
        current_animation_time++;
        if (scene_filename == "fsa") {
            hm.set_hm(wave_function);
            // Free last object used in this line... <--------------------------------------------------------------------------- ADDRESS THIS!!!
            objects[0] = hm.get_hm_obj();
        }
    }
}

/* The 'main' function:
 *
 * Run the OpenGL program (initialize OpenGL, display the scene, and react to
 * mouse and keyboard presses).
 */
int main(int argc, char* argv[]) {
    if (argc != 5) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [scene_description_file.txt]"
                                 + " [shader_mode]"
                                 + " [xres]"
                                 + " [yres]";
        cout << usage_statement << endl;
        return 1;
    }

    scene_filename = argv[1];
    if (scene_filename == "../data/scenes/scene_heavy.txt"
        || scene_filename == "../data/scenes/scene_square.txt") {
        create_texture_coords_buffer = true;
        generate_normals = false;
        // generate_normals = true;
    }
    else {
        create_texture_coords_buffer = false;
        generate_normals = true;
    }

    shader_mode = argv[2];

    xres = atoi(argv[3]); // 700-800 is good
    yres = atoi(argv[4]); // 700-800 is good

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(xres, yres);

    glutInitWindowPosition(0, 0);

    glutCreateWindow("Illustrative Rendering");

    if (shader_mode == "sc") { // solid color without GLSL shaders
        // generate_normals = true;
        cout << "Shader Mode is solid color OpenGL smooth (Gouraud) shading)"
             << endl;
        create_texture_coords_buffer = false;
        use_textures = false;
    }
    else if (shader_mode == "tf2") { // Team Fortress 2 Artstyle
        cout << "Shader Mode is Team Fortress 2 Artstyle" << endl;
        vertProgFileName = "shaders/vs_texturing.glsl";
        fragProgFileName = "shaders/fs_texturing_tf2_artstyle.glsl";
    }
    else if (shader_mode == "tex_cs") { // texture cell shading
        cout << "Shader Mode is texture cel shading" << endl;
        vertProgFileName = "shaders/vs_texturing.glsl";
        fragProgFileName = "shaders/fs_texture_cel_shading.glsl";
    }
    else if (shader_mode == "sc_glsl") { // solid color with GLSL shaders
        cout << "Shader Mode is solid color GLSL Phong shading)" << endl;
        vertProgFileName = "shaders/vs_solid_color.glsl";
        fragProgFileName = "shaders/fs_solid_color.glsl";
    }
    else if (shader_mode == "tex_phong") { // GLSL phong shaders
        cout << "Shader Mode is GLSL Phong shading" << endl;
        vertProgFileName = "shaders/vs_texturing.glsl";
        fragProgFileName = "shaders/fs_texturing_phong.glsl";
    }
    else {
        vertProgFileName = "shaders/vs_" + shader_mode + ".glsl";
        fragProgFileName = "shaders/fs_" + shader_mode + ".glsl";
    }

    init(scene_filename);

    glutIdleFunc(idle);

    glutDisplayFunc(display);

    glutReshapeFunc(reshape);

    glutMouseFunc(mouse_pressed);

    glutMotionFunc(mouse_moved);

    glutKeyboardFunc(key_pressed);

    glutMainLoop();
}
