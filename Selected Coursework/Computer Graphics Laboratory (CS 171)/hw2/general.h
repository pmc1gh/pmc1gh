// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// general.h

#ifndef general_h
#define general_h

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>

#include "camera.h"
#include "light.h"
#include "object.h"

using namespace std;
using namespace Eigen;

/**
 * Return true if a given string is found in a vector of strings. Otherwise
 * return false.
 */
bool string_in_vector(const vector<string> &v, const string a);

/**
 * Return the index of a given string in a vector of strings if the given string
 * is found. Otherwise, return -1 if the given string is not found in the
 * vector.
 */
int string_index_in_vector(const vector<string> &v, const string a);

#endif
