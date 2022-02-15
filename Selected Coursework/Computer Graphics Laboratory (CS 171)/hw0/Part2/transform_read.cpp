// Philip Carr
// CS 171 Assignment 0 Part 2
// October 10, 2018
// transform_read.cpp

#include <stdio.h>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <math.h>
#include <assert.h>

using namespace std;
using namespace Eigen;

/**
 * Return a vector of all the matrix transformations in a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if neither "t", "r", nor "s" is found at beginning of
 * a non-empty line.
 */
vector<Matrix4d> read_transforms(const string filename) {
    ifstream ifs;
    // Open the file to read the transformations
    ifs.open(filename);

    // Make sure the file was opened successfully
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    vector<Matrix4d> transform_vector;

    while (true) {
        string d_type;
        ifs >> d_type;

        // Check if done reading file
        if (ifs.eof()) {
            break;
        }

        // Create translation matrix if "t" found
        if (d_type == "t") {
            double tx;
            double ty;
            double tz;
            ifs >> tx >> ty >> tz;
            Matrix4d new_matrix;
            new_matrix << 1, 0, 0, tx,
                          0, 1, 0, ty,
                          0, 0, 1, tz,
                          0, 0, 0, 1;
            transform_vector.push_back(new_matrix);
        }

        // Create rotation matrix if "r" found
        else if (d_type == "r") {
            double rx;
            double ry;
            double rz;
            double angle;
            ifs >> rx >> ry >> rz >> angle;
            double a00 = rx * rx + (1 - rx * rx) * cos(angle);
            double a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
            double a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
            double a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
            double a11 = ry * ry + (1 - ry * ry) * cos(angle);
            double a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
            double a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
            double a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
            double a22 = rz * rz + (1 - rz * rz) * cos(angle);
            Matrix4d new_matrix;
            new_matrix << a00, a01, a02, 0,
                          a10, a11, a12, 0,
                          a20, a21, a22, 0,
                          0, 0, 0, 1;
            transform_vector.push_back(new_matrix);
        }

        // Create scaling matrix if "s" found
        else if (d_type == "s") {
            double sx;
            double sy;
            double sz;
            ifs >> sx >> sy >> sz;
            Matrix4d new_matrix;
            new_matrix << sx, 0, 0, 0,
                          0, sy, 0, 0,
                          0, 0, sz, 0,
                          0, 0, 0, 3;
            transform_vector.push_back(new_matrix);
        }

        /* Throw error if neither "t", "r", nor "s" is found at beginning of
         * line
         */
        else {
            string error_message = (string) "txt file data type is neither "
                                   + "vertex nor value";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file
    ifs.close();

    return transform_vector;
}

/**
 * Print the inverse transformation of the product of transformations in a given
 * vector of transformations.
 */
void print_inverse_transform(const string filename,
                             const vector<Matrix4d> &transform_vector) {
    Matrix4d transform;
    transform << 1, 0, 0, 0,
                 0, 1, 0, 0,
                 0, 0, 1, 0,
                 0, 0, 0, 1;

    // Multiply all the transformations together into one transformation
    for (int i = 0; i < (int) transform_vector.size(); i++) {
        transform = transform_vector[i] * transform;
    }

    cout << filename << ":" << endl;
    cout << "Inverse transform:\n" << endl;
    Matrix4d inv_m = transform.inverse();
    cout << inv_m << endl;
    cout << endl;
}

int main(int argc, char *argv []) {
    // Check to make sure there is exactly one command line argument
    if (argc != 2) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " transform_dataN.txt";
        cout << usage_statement << endl;
        return 1;
    }
    vector<Matrix4d> transform_vector;
    transform_vector = read_transforms(argv[1]);
    print_inverse_transform(argv[1], transform_vector);
    return 0;
}
