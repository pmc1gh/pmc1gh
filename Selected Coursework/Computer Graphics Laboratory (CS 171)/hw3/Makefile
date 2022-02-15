# Philip Carr
# CS 171 Assignment 3
# November 2, 2018
# Makefile

CC = g++
FLAGS = -g -std=c++11 -o

INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES = *.cpp *.h
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = opengl_renderer

all: $(SOURCES)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
