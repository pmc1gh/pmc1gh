# Philip Carr
# CS 171 Assignment 6 Part 1
# November 30, 2018
# Makefile

CC = g++
FLAGS = -g -std=c++11

INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -isystem../../
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES = *.h *.cpp
LIBS = -lGLEW -lGL -lGLU -lglut -lm

EXENAME = simulate

all: $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
