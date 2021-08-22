// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// vs_texturing.glsl

attribute vec3 tangent;
// attribute float tangent;

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
    tangent_copy = normalize(tangent); // tangents must be normalized

    // Renormalize the tangent vector with respect to the object normal vector.
    tangent_copy = normalize(tangent_copy - dot(tangent_copy, obj_n) * obj_n);

    // Transform points using the model view projecttion matrix
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

    // Get the current texture coordinates.
    gl_TexCoord[0] = gl_MultiTexCoord0;
}
