# Philip Carr
# CS 171 Assignment 2
# October 26, 2018
# Makefile

CC = g++
FLAGS = -g -std=c++11 -O2 # O2 is a compiler optimization, also could use -O3

# The following line is a relative directory reference that assumes the Eigen
# folder--which your program will depend on--is located one directory above the
# directory that contains this Makefile.
INCLUDE = -isystem../
SOURCES = *.h *.cpp

EXENAME = shaded_renderer

SCENE_NAMES = scene_armadillo scene_bunny1_attenuated scene_bunny1 \
	scene_bunny2 scene_cube1 scene_cube2 scene_cube3 \
	scene_fourCubes scene_kitten_attenuated scene_kitten \
	scene_sphere_attenuated scene_sphere

all: $(EXENAME)

$(EXENAME): $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(INCLUDE) $(SOURCES)

g_images: $(EXENAME)
	for name in $(SCENE_NAMES) ; do \
		./$(EXENAME) data/$${name}.txt 800 800 0 \
		| convert - $${name}_Gouraud.png; \
	done

p_images: $(EXENAME)
	for name in $(SCENE_NAMES) ; do \
		./$(EXENAME) data/$${name}.txt 800 800 1 \
		| convert - $${name}_Phong.png ; \
	done

clean:
	rm -f *.o $(EXENAME) *.png

.PHONY: all clean
