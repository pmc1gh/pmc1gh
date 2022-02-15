// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// script_read.cpp

#include "script_read.h"

using namespace std;
using namespace Eigen;

/**
 * Print the script data in a way similar to the way the .script files are
 * organized.
 */
void print_script_data(Script_Data sd) {
    cout << "Script Data:\n" << endl;
    cout << "Total number of frames: " << sd.n_frames << "\n" << endl;
    for (int i = 0; i < (int) sd.keyframes.size(); i++) {
        Keyframe kf = sd.keyframes[i];
        cout << "Frame " << kf.frame_number << endl;
        cout << "translation " << kf.translation(0) << " " << kf.translation(1)
             << " " << kf.translation(2) << endl;
        cout << "scale " << kf.scale(0) << " " << kf.scale(1) << " "
             << kf.scale(2) << endl;
        cout << "rotation " << kf.rotation.get_v()[0] << " "
             << kf.rotation.get_v()[1] << " " << kf.rotation.get_v()[2] << " "
             << kf.rotation.get_s() << endl;
    }
}

/**
 * Return a KeyFrame struct as read from a script file.
 */
Script_Data read_script_file(const string filename) {
    ifstream ifs;

    // Open the file to read the object.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    int n_frames;
    ifs >> n_frames;

    Script_Data script_data;

    script_data.n_frames = n_frames;

    int frame_number;
    Vector3f translation;
    Vector3f scale;
    quaternion rotation;

    int count_params_read = 0;
    while (true) {
        string d_type;
        float a;
        float b;
        float c;
        float d;
        ifs >> d_type;

        if (count_params_read == 4) {
            Keyframe keyframe;
            keyframe.frame_number = frame_number;
            keyframe.translation = translation;
            keyframe.scale = scale;
            keyframe.rotation = rotation;
            script_data.keyframes.push_back(keyframe);
            count_params_read = 0;
        }

        // Check if done reading file.
        if (ifs.eof()) {
            break;
        }

        // Add new vertex to vertex vector if "v" found at beginning of line.
        if (d_type == "Frame") {
            ifs >> frame_number;
            count_params_read++;
        }

        /* Initialize translation vector if "translation" found at beginning of
         * line.
         */
        else if (d_type == "translation") {
            ifs >> a >> b >> c;
            translation(0) = a;
            translation(1) = b;
            translation(2) = c;
            count_params_read++;
        }

        /* Initialize scale vector if "scale" found at beginning of line.
         */
        else if (d_type == "scale") {
            ifs >> a >> b >> c;
            scale(0) = a;
            scale(1) = b;
            scale(2) = c;
            count_params_read++;
        }

        /* Initialize translation vector if "rotation" found at beginning of
         * line.
         */
        else if (d_type == "rotation") {
            ifs >> a >> b >> c >> d;
            float mag = sqrt(a * a + b * b + c * c);
            a /= mag;
            b /= mag;
            c /= mag;
            d *= M_PI / 180.0;
            rotation = quaternion(cos(d / 2), a * sin(d / 2), b * sin(d / 2),
                                  c * sin(d / 2));
            count_params_read++;
        }

        /* Throw error if neither "Frame", "translation", "scale" nor "rotation"
         * is found at beginning of line.
         */
        else {
            string error_message = (string) "script file data type is neither "
                                   + "Frame number, translation, scale, nor "
                                   + "rotation";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file.
    ifs.close();

    return script_data;
}
