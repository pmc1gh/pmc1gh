// Philip Carr
// CS 171 Assignment 6 Part 1
// November 30, 2018
// utils.h

// Note: This file was originally saved on December 1, 2018, at 1:18:31am.
// The only modification made to this file since then is that lines 45, 56, 57,
// 58, and 59 were commented out to avoid conflict with the c++'s own float abs
// function.
//
// The compilation error that previously occurred when running make:
// g++ -g -std=c++11 -o simulate -I/usr/X11R6/include -I/usr/include/GL
// -I/usr/include -isystem../../ -L/usr/X11R6/lib -L/usr/local/lib *.h *.cpp -lGLEW
// -lGL -lGLU -lglut -lm
// In file included from elastic_demo.cpp:19:
// utils.h:23:25: error: ‘float abs(float)’ conflicts with a previous declaration
//    23 | static float abs(float n);
//       |                         ^
// In file included from /usr/include/c++/9/cstdlib:77,
//                  from /usr/include/c++/9/stdlib.h:36,
//                  from /usr/include/GL/freeglut_std.h:608,
//                  from /usr/include/GL/glut.h:17,
//                  from elastic_demo.cpp:8:
// /usr/include/c++/9/bits/std_abs.h:75:3: note: previous declaration ‘constexpr
// float std::abs(float)’
//    75 |   abs(float __x)
//       |   ^~~
// make: *** [Makefile:17: all] Error 1
//
// Date and time of this note: October 30, 2020, at 3:05pm.

/* CS/CNS 171
 * Fall 2015
 * Written by Kevin (Kevli) Li (Class of 2016)
 *
 * Straightforward utility functions.
 */

#ifndef UTILS_H
#define UTILS_H

#include <sstream>
#include <string>
#include <vector>

/* Function prototypes */

/* Absolute value function */
// static float abs(float n);

/* Splits given string based on given delimiter and stores tokens in given vector */
static std::vector<std::string> &split(const std::string &str,
                                       char delim,
                                       std::vector<std::string> &tokens);
/* Splits given string based on given delimiter and returns tokens in a vector */
static std::vector<std::string> split(const std::string &str, char delim);

/* Function implementations */

// static float abs(float n)
// {
//     return (n < 0) ? n * -1.0 : n;
// }

static std::vector<std::string> &split(const std::string &str,
                                       char delim,
                                       std::vector<std::string> &tokens)
{
    std::stringstream ss(str);
    std::string token;

    while (getline(ss, token, delim))
        if(!token.empty())
            tokens.push_back(token);

    return tokens;
}

static std::vector<std::string> split(const std::string &str, char delim)
{
    std::vector<std::string> tokens;
    split(str, delim, tokens);
    return tokens;
}

#endif
