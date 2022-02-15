// Philip Carr
// CS 171 Assignment 2
// October 26, 2018
// general.cpp

#include "general.h"

using namespace std;

/**
 * Return true if a given string is found in a vector of strings. Otherwise
 * return false.
 */
bool string_in_vector(const vector<string> &v, const string a) {
    for (int i = 0; i < (int) v.size(); i++) {
        if (v[i] == a) {
            return true;
        }
    }
    return false;
}

/**
 * Return the index of a given string in a vector of strings if the given string
 * is found. Otherwise, return -1 if the given string is not found in the
 * vector.
 */
int string_index_in_vector(const vector<string> &v, const string a) {
    for (int i = 0; i < (int) v.size(); i++) {
        if (v[i] == a) {
            return i;
        }
    }
    return -1;
}
