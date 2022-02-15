# Philip Carr
# CS 171 Assignment 6 Part 1
# November 30, 2018
# Makefile

CC = g++
FLAGS = -g

INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES = *.cpp
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = single_pendulum

all: $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
