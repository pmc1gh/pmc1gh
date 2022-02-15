// Philip Carr
// CS 171 Assignment 4 Part 2
// November 17, 2018
// fragmentProgram.glsl

// Total number of lights in scene.
uniform int n_lights;
uniform sampler2D texture_image, texture_normals;

// Object vertex.
varying vec4 obj_v;

// Object vertex normal vector.
varying vec3 obj_n;

varying vec3 tangent_copy;

void main()
{
    // Object vertex in surface coordinates.
    vec3 obj_vs;

    /* Compute the bitangent vector and object vertex coordinates in surface
     * coordinates.
     */
    vec3 bitangent = cross(obj_n, tangent_copy);
    obj_vs = vec3(dot(tangent_copy, obj_v.xyz), dot(bitangent, obj_v.xyz),
                  dot(obj_n, obj_v.xyz));

    // Get the current object lighting properties.
    vec4 ambient = 0.1 * texture2D(texture_image, gl_TexCoord[0].st);
    vec4 diffuse = texture2D(texture_image, gl_TexCoord[0].st);
    vec4 specular = texture2D(texture_image, gl_TexCoord[0].st);
    float shininess = gl_FrontMaterial.shininess;

    /* Get the current texture normal vector from the normal vector texture
     * file.
     */
    vec3 normal = texture2D(texture_normals, gl_TexCoord[0].st).rgb;

    /* Transform the vector components from [0, 1] to [-1, 1] and then normalize
     * the normal vector.
     */
    normal = normal * 2.0 - 1.0;
    normal = normalize(normal);

    vec4 diffuse_sum = vec4(0, 0, 0, 0);
    vec4 specular_sum = vec4(0, 0, 0, 0);

    /* Camera is at origin in camera space, so camera direction is just the
     * negative of the object vertex coordinates.
     */
    vec3 cam_dir = -obj_vs.xyz;
    cam_dir = normalize(cam_dir);

    // Implement the Phong lighting model.
    for(int i = 0; i < n_lights; i++) {
        vec3 light_vs;
        vec3 light_pos = gl_LightSource[i].position.xyz;

        // Compute the light position in surface coordinates.
        light_vs = vec3(dot(tangent_copy, light_pos), dot(bitangent, light_pos),
                        dot(obj_n, light_pos));

        vec4 light_diffuse = gl_LightSource[i].diffuse;
        vec4 light_specular = gl_LightSource[i].specular;
        vec3 l_dir = (light_vs - obj_vs).xyz;
        float dist_squared = dot(l_dir, l_dir);

        float light_k = gl_LightSource[i].quadraticAttenuation;

        float color_denom = 1.0 + light_k * dist_squared;
        light_diffuse /= color_denom;
        light_specular /= color_denom;
        l_dir = normalize(l_dir);

        float diff_val = max(0.0, dot(normal, l_dir));
        vec4 l_diff = diff_val * light_diffuse;
        diffuse_sum += l_diff;

        vec3 dir_sum = l_dir + cam_dir;
        float dot_p2 = dot(normal, normalize(dir_sum));
        float spec_val = pow(max(0.0, dot_p2), shininess);
        vec4 l_spec = spec_val * light_specular;
        specular_sum += l_spec;
    }

    vec4 color = min(vec4(1,1,1,1), ambient + diffuse * diffuse_sum
                     + specular * specular_sum);

    gl_FragColor = color;
}
