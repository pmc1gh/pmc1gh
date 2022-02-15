// Philip Carr
// CS 171 Assignment 4 Part 2
// November 17, 2018
// vertexProgram.glsl

attribute vec3 tangent;

// Object vertex.
varying vec4 obj_v;

// Object normal vector
varying vec3 obj_n;

/* Tangent vector copy to be used in the fragment shader.
 * Note: Camera coordinates to surface coordinates transformations were
 * originally planned to be performed here for each vertex, but I decided to
 * perform these transformations in the fragment shaders to allow for the
 * possible use of multiple lights in the scene.
 */
varying vec3 tangent_copy;

void main()
{
    /* Transform the vertex and normal vector by their respective
     * transformations in world space for use in the fragment shader.
     */
    obj_v = gl_ModelViewMatrix * gl_Vertex;
    obj_n = normalize(gl_NormalMatrix * gl_Normal);

    // Set the tangent vector copy equal to the original tangent vector.
    tangent_copy = tangent;

    // Transform points using the model view projecttion matrix
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    // Get the current texture coordinates.
    gl_TexCoord[0] = gl_MultiTexCoord0;
}
