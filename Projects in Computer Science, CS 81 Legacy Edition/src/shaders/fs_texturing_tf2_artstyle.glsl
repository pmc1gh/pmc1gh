// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// fs_texturing_tf2_artstyle.glsl

// Total number of lights in scene.
uniform int n_lights;
uniform sampler2D texture_image, texture_normals, warped_diffuse_texture;

// Object vertex.
varying vec4 obj_v;

// Object vertex normal vector.
varying vec3 obj_n;

varying vec3 tangent_copy;

/**
 * Compute the ambient cube color as seen in the Team Fortress 2 Artstyle paper.
 * This function was hard-coded due to time constraints. If this is to be fully
 * implemented, the ambient lighting parameters of in-game scene shown in the
 * paper would have to be extracted for the ambient cube parameters to be
 * computed.
 */
vec3 ambient_cube_color(const vec3 world_normal, bool renormalize) {
    vec3 normal_squared;
    if (renormalize) {
        vec3 renormalized_world_normal = normalize(world_normal);
        normal_squared = renormalized_world_normal * renormalized_world_normal;
    }
    else {
        normal_squared = world_normal * world_normal;
    }
    // vec3 normal_squared = world_normal;
    bvec3 is_negative = bvec3(world_normal.x < 0.0, world_normal.y < 0.0,
                              world_normal.z < 0.0);
    float linear_color = 0.0;
    if (is_negative.x) {
        linear_color += normal_squared.x * 0.3;
    }
    else {
        linear_color += normal_squared.x * 0.3;
    }
    if (is_negative.y) {
        linear_color += normal_squared.y * 0.17;
    }
    else {
        linear_color += normal_squared.y * 0.58;
    }
    if (is_negative.z) {
        linear_color += normal_squared.z * 0.3;
    }
    else {
        linear_color += normal_squared.z * 0.3;
    }
    return vec3(linear_color, linear_color, 1.1 * linear_color);
}

// Return the inverse of a 3x3 matrix.
mat3 get_inverse3x3(mat3 m) {
    float det = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
                - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
                + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
    float inv00 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
    float inv01 = m[2][1] * m[0][2] - m[0][1] * m[2][2];
    float inv02 = m[0][1] * m[1][2] - m[1][1] * m[0][2];

    float inv10 = m[2][0] * m[1][2] - m[1][0] * m[2][2];
    float inv11 = m[0][0] * m[2][2] - m[2][0] * m[0][2];
    float inv12 = m[1][0] * m[0][2] - m[0][2] * m[1][2];

    float inv20 = m[1][0] * m[2][1] - m[2][0] * m[1][1];
    float inv21 = m[2][0] * m[0][1] - m[0][0] * m[2][1];
    float inv22 = m[0][0] * m[1][1] - m[1][0] * m[0][1];

    mat3 inv_matrix =
        mat3(vec2(inv00, inv01), inv02,
             vec2(inv10, inv11), inv12,
             vec2(inv20, inv21), inv22) / det;

    return inv_matrix;
}

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
    vec4 specular =  1.0 * texture2D(texture_image, gl_TexCoord[0].st);
    float shininess = gl_FrontMaterial.shininess;

    vec4 albedo_color = texture2D(texture_image, gl_TexCoord[0].st);

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

    float fr = pow(1.0 - max(0.0, dot(normal, cam_dir)), 2.0);
    float kr = 3.0;

    // Implement the specialized lighting model.
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

        // Diffuse lighting
        float diff_val = max(0.0, dot(normal, l_dir));

        // Diffuse Lighting (Illustrative Rendering)
        float alpha = 0.5;
        float beta = 0.5;
        float gamma = 1.0;
        float wda = pow(alpha * max(0.0, dot(normal, l_dir)) + beta, gamma);
        vec4 wd_term = texture2D(warped_diffuse_texture, vec2(wda, 0.5));
        vec4 wd_color = light_diffuse * wd_term;
        diffuse_sum += wd_color;

        // Specular lighting
        vec3 dir_sum = l_dir + cam_dir;
        float dot_p2 = dot(normal, normalize(dir_sum));

        // Specular lighting (Illustrative Rendering)
        float ks = 0.5;
        float fs = 1.0;
        float kspec = 10.0; // this is equivalent to shininess
        float krim = 1.0;
        float spec_val = fs * pow(max(0.0, dot_p2), shininess);
        float rim_val1 = fr * kr * pow(max(0.0, dot_p2), krim);
        vec4 l_spec = ks * max(spec_val, rim_val1) * light_specular;
        specular_sum += l_spec;
    }

    // Take the inverse of the TBN matrix and multiply it to normal to get the
    // normal vector from texture space into world space.
    vec3 normalized_obj_n = normalize(obj_n);
    mat3 tbn_matrix =
        mat3(vec2(tangent_copy.x, bitangent.x), normalized_obj_n.x,
             vec2(tangent_copy.y, bitangent.y), normalized_obj_n.y,
             vec2(tangent_copy.z, bitangent.z), normalized_obj_n.z);
    mat3 inverse_tbn_matrix = get_inverse3x3(tbn_matrix);
    vec3 normal_ws = inverse_tbn_matrix * normal;
    vec3 acc = ambient_cube_color(normal_ws, true);

    diffuse_sum += vec4(acc.x, acc.y, acc.z, 1);

    vec3 u = vec3(0.0, 1.0, 0.0);
    vec3 u_vs = vec3(dot(tangent_copy, u), dot(bitangent, u), dot(obj_n, u));
    vec3 av = ambient_cube_color(-obj_v.xyz, true);
    vec4 rim_val2 = dot(normal, u_vs) * fr * kr * vec4(av.x, av.y, av.z, 1);

    /**
     * To view specific components of the Illustrative Rendering, uncomment
     * only the desired lines of the vec4 color assignment:
     */

    // Albedo color
    // vec4 color = min(vec4(1,1,1,1), albedo_color);

    // Warped diffuse color only
    // diffuse_sum -= vec4(acc.x, acc.y, acc.z, 1);
    // vec4 color = min(vec4(1,1,1,1), diffuse_sum);

    // Ambient cube color only
    // vec4 color = vec4(acc.x, acc.y, acc.z, 1);

    // Warped diffuse + ambient cube color
    // vec4 color = min(vec4(1,1,1,1), diffuse_sum);

    // Specular color only
    // vec4 color = min(vec4(1,1,1,1), specular_sum);

    // Dedicated rim lighting only
    // vec4 color = min(vec4(1,1,1,1), rim_val2);

    // Specular + dedicated rim lighting term
    // vec4 color = min(vec4(1,1,1,1), specular_sum + rim_val2);

    // Full color
    vec4 color = min(vec4(1,1,1,1),
                     albedo_color * diffuse_sum
                     + 0.5 * (specular_sum + rim_val2));

    gl_FragColor = color;
}
