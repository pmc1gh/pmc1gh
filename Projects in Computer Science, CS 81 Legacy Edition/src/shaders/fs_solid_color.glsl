// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// fs_solid_color.glsl

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
    // Get the current object lighting properties.
    vec4 ambient = gl_FrontMaterial.ambient;
    vec4 diffuse = gl_FrontMaterial.diffuse;
    vec4 specular = gl_FrontMaterial.specular;
    float shininess = gl_FrontMaterial.shininess;

    vec4 diffuse_sum = vec4(0, 0, 0, 0);
    vec4 specular_sum = vec4(0, 0, 0, 0);

    /* Camera is at origin in camera space, so camera direction is just the
     * negative of the object vertex coordinates.
     */
    vec3 cam_dir = -obj_v.xyz;
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
        vec3 l_dir = (light_pos - obj_v.xyz).xyz;
        float dist_squared = dot(l_dir, l_dir);

        float light_k = gl_LightSource[i].quadraticAttenuation;

        float color_denom = 1.0 + light_k * dist_squared;
        light_diffuse /= color_denom;
        light_specular /= color_denom;
        l_dir = normalize(l_dir);

        float diff_val = max(0.0, dot(obj_n, l_dir));
        vec4 l_diff = diff_val * light_diffuse;
        diffuse_sum += l_diff;

        vec3 dir_sum = l_dir + cam_dir;
        float dot_p2 = dot(obj_n, normalize(dir_sum));
        float spec_val = pow(max(0.0, dot_p2), shininess);
        vec4 l_spec = spec_val * light_specular;
        specular_sum += l_spec;
    }

    vec4 color = min(vec4(1,1,1,1), ambient + diffuse * diffuse_sum
                     + specular * specular_sum);

    gl_FragColor = color;
}
