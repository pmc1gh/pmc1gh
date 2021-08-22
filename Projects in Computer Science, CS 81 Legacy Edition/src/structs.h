// Philip Carr
// CS 81ab Project: Illustrative Rendering
// CS 81c Project: Fluid Surface Animation
// October 28, 2020
// structs.h

#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>

struct Vec3f
{
	float x, y, z;
};

struct Vertex
{
    float x, y, z;
};

struct TextureCoords {
    float u;
    float v;
};

struct Face
{
	int vidx1, vidx2, vidx3;
	int vtidx1, vtidx2, vtidx3;
    int vnidx1, vnidx2, vnidx3;
};

struct Mesh_Data
{
	std::vector<Vertex*> *vertices;
	std::vector<TextureCoords*> *vertex_texture_coords;
    std::vector<Vec3f*> *vertex_normals;
    std::vector<Face*> *faces;
};

#endif
