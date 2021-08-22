// CS 179 Project: GPU-accelerated ray tracing
// Team members: Philip Carr and Thomas Leing
// June 10, 2019
// raytrace.hpp

#include <png.h>

#include "Utilities.hpp"

class PNGMaker {
    public:
        int xres;
        int yres;

        PNGMaker(int xres, int yres);
        ~PNGMaker();

        void setPixel(int x, int y, float r, float g, float b);
        int saveImage();

    private:
        Vec3f_Ut *screen;
};
