// Philip Carr
// CS 171 Assignment 6 Part 2
// November 30, 2018
// quaternion.cpp

#include "quaternion.h"

using namespace std;

// quaternion class member function definitions.

// Default constructor for a quaternion.
quaternion::quaternion() {
    s = 0;
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
}

// 4-argument constructor for a quaternion.
quaternion::quaternion(const float s, const float v0, const float v1,
                       const float v2) {
    this->s = s;
    this->v[0] = v0;
    this->v[1] = v1;
    this->v[2] = v2;
}

// Return the quaternion's s-value.
float quaternion::get_s() const {
    return this->s;
}

// Return the quaternion's v-vector.
float *quaternion::get_v() {
    return this->v;
}

// Set the value of s of a quaternion.
void quaternion::set_s(const float s) {
    this->s = s;
}

// Set the vector v of a quaternion.
void quaternion::set_v(const float v0, const float v1, const float v2) {
    this->v[0] = v0;
    this->v[1] = v1;
    this->v[2] = v2;
}

// Add a quaternion q to the given quaternion.
void quaternion::add(quaternion &q) {
    float qs = q.get_s();
    float *qv = q.get_v();
    float qx = qv[0];
    float qy = qv[1];
    float qz = qv[2];

    this->s += qs;
    this->v[0] += qx;
    this->v[1] += qy;
    this->v[2] += qz;
}

// Multiply a scalar s to the given quaternion.
void quaternion::s_mult(const float s) {
    this->s *= s;
    this->v[0] *= s;
    this->v[1] *= s;
    this->v[2] *= s;
}

// Return the product of the quaternion q and the given quaternion.
quaternion quaternion::q_mult(quaternion &q) const { // Use Eigen
    float qs = q.get_s();
    float *qv = q.get_v();
    float qx = qv[0];
    float qy = qv[1];
    float qz = qv[2];

    float mult_s = this->s * qs - this->v[0] * qx - this->v[1] * qy
                   - this->v[2] * qz;
    float mult_v0 = this->s * qx + this->v[0] * qs + this->v[1] * qz
                    - this->v[2] * qy;
    float mult_v1 = this->s * qy + this->v[1] * qs + this->v[2] * qx
                    - this->v[0] * qz;
    float mult_v2 = this->s * qz + this->v[2] * qs + this->v[0] * qy
                    - this->v[1] * qx;

    return quaternion(mult_s, mult_v0, mult_v1, mult_v2);
}

// Return the complex conjugate of the given quaternion.
quaternion quaternion::conj() const {
    float neg_v0 = -this->v[0];
    float neg_v1 = -this->v[1];
    float neg_v2 = -this->v[2];
    return quaternion(this->s, neg_v0, neg_v1, neg_v2);
}

// Return the norm of the given quaternion (sqrt(s^2 + x^2 + y^2 + z^2)).
float quaternion::norm() const {
    return sqrt(this->s * this->s + this->v[0] * this->v[0]
           + this->v[1] * this->v[1] + this->v[2] * this->v[2]);
}

// Return the inverse of the given quaternion (q^*/(|q|^2)).
quaternion quaternion::inv() const {
    float norm = this->norm();
    quaternion q_star_normalized = this->conj();
    q_star_normalized.set_s(q_star_normalized.get_s() / (norm * norm));
    float *v = q_star_normalized.get_v();
    q_star_normalized.set_v(v[0] / (norm * norm), v[1] / (norm * norm),
                            v[2] / (norm * norm));
    return q_star_normalized;
}

// Normalize the given quaternion.
void quaternion::normalize() {
    float norm = this->norm();
    this->s = this->s / norm;
    this->v[0] = this->v[0] / norm;
    this->v[1] = this->v[1] / norm;
    this->v[2] = this->v[2] / norm;
}
