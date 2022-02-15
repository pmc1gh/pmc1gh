# Philip Carr
# CS 171 Assignment 0 Part 4
# October 10, 2018
# Makefile

CC = g++
FLAGS = -g -Wall -std=c++11
SOURCES = *.h *.cpp

EXENAME = circle_image

all: $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(INCLUDE) $(SOURCES)

clean:
	rm -f *.o *.ppm $(EXENAME)

.PHONY: all clean
