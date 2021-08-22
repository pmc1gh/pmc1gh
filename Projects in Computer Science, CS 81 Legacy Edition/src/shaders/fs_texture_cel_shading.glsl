// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// fs_texture_cel_shading.glsl

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

        float diff_val = max(0.3, dot(normal, l_dir));
        vec4 l_diff = diff_val * light_diffuse;
        diffuse_sum += l_diff;

        vec3 dir_sum = l_dir + cam_dir;
        float dot_p2 = dot(normal, normalize(dir_sum));
        float spec_val = pow(max(0.3, dot_p2), shininess);
        vec4 l_spec = spec_val * light_specular;
        specular_sum += l_spec;
    }

    vec4 color = min(vec4(1,1,1,1), ambient + diffuse * diffuse_sum
                     + specular * specular_sum);

    // cel shading routine
    float num_cel_shaded_colors = 3.0;
    float red = color.x;
    float green = color.y;
    float blue = color.z;
    float max_rgb_value = max(max(red, green), blue);
    float min_rgb_value = min(min(red, green), blue);
    float lightness = (max_rgb_value + min_rgb_value) / 2.0;

    float saturation = 0.0;
    float red_diff = color.x;
    float green_diff = color.y;
    float blue_diff = color.z;
    float max_rgb_value_diff = max(max(red_diff, green_diff), blue_diff);
    float min_rgb_value_diff = min(min(red_diff, green_diff), blue_diff);
    if (max_rgb_value_diff != min_rgb_value_diff) {
        if (lightness < 0.5) {
            saturation = (max_rgb_value_diff - min_rgb_value_diff)
                         / (max_rgb_value_diff + min_rgb_value_diff);
        }
        else {
            saturation = (max_rgb_value_diff - min_rgb_value_diff)
                         / (2.0 - max_rgb_value_diff - min_rgb_value_diff);
        }
    }

    float hue;
    if (red_diff == max_rgb_value_diff) {
        hue = (green_diff - blue_diff)
              / (max_rgb_value_diff - min_rgb_value_diff);
    }
    else if (green_diff == max_rgb_value_diff) {
        hue = 2.0 + (blue_diff - red_diff)
                    / (max_rgb_value_diff - min_rgb_value_diff);
    }
    else {
        hue = 4.0 + (red_diff - green_diff)
                    / (max_rgb_value_diff - min_rgb_value_diff);
    }
    hue *= 60.0;
    if (hue < 0.0) {
        hue += 360.0;
    }

    float cel_shaded_lightness = 0.0;
    while (cel_shaded_lightness < lightness) {
        cel_shaded_lightness += 1.0 / num_cel_shaded_colors;
    }

    // hsl to rgb conversion
    float red_cel_shaded, green_cel_shaded, blue_cel_shaded;
    if (saturation == 0.0) {
        red_cel_shaded = cel_shaded_lightness;
        green_cel_shaded = cel_shaded_lightness;
        blue_cel_shaded = cel_shaded_lightness;
    }
    else {
        float temp1, temp2;
        if (cel_shaded_lightness < 0.5) {
            temp1 = cel_shaded_lightness * (1.0 + saturation);
        }
        else {
            temp1 = cel_shaded_lightness + saturation
                    - cel_shaded_lightness * saturation;
        }

        temp2 = 2.0 * cel_shaded_lightness - temp1;

        hue /= 360.0;

        float temp_red, temp_green, temp_blue;
        temp_red = hue + 1.0 / 3.0;
        temp_green = hue;
        temp_blue = hue - 1.0 / 3.0;

        if (temp_red < 0.0) {
            temp_red += 1.0;
        }
        if (temp_green < 0.0) {
            temp_green += 1.0;
        }
        if (temp_blue < 0.0) {
            temp_blue += 1.0;
        }

        if (temp_red > 1.0) {
            temp_red -= 1.0;
        }
        if (temp_green > 1.0) {
            temp_green -= 1.0;
        }
        if (temp_blue > 1.0) {
            temp_blue -= 1.0;
        }

        if (6.0 * temp_red < 1.0) {
            red_cel_shaded = temp2 + (temp1 - temp2) * 6.0 * temp_red;
        }
        else if (2.0 * temp_red < 1.0) {
            red_cel_shaded = temp1;
        }
        else if (3.0 * temp_red < 2.0) {
            red_cel_shaded = temp2 + (temp1 - temp2) * (2.0 / 3.0 - temp_red)
                             * 6.0;
        }
        else {
            red_cel_shaded = temp2;
        }

        if (6.0 * temp_green < 1.0) {
            green_cel_shaded = temp2 + (temp1 - temp2) * 6.0 * temp_green;
        }
        else if (2.0 * temp_green < 1.0) {
            green_cel_shaded = temp1;
        }
        else if (3.0 * temp_green < 2.0) {
            green_cel_shaded = temp2 + (temp1 - temp2)
                                  * (2.0 / 3.0 - temp_green) * 6.0;
        }
        else {
            green_cel_shaded = temp2;
        }

        if (6.0 * temp_blue < 1.0) {
            blue_cel_shaded = temp2 + (temp1 - temp2) * 6.0 * temp_blue;
        }
        else if (2.0 * temp_blue < 1.0) {
            blue_cel_shaded = temp1;
        }
        else if (3.0 * temp_blue < 2.0) {
            blue_cel_shaded = temp2 + (temp1 - temp2)
                                 * (2.0 / 3.0 - temp_blue) * 6.0;
        }
        else {
            blue_cel_shaded = temp2;
        }
    }

    vec4 color_cel_shaded =
        vec4(red_cel_shaded, green_cel_shaded, blue_cel_shaded, 1);

    gl_FragColor = color_cel_shaded;
}