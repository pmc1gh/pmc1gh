# Philip Carr
# CS 171 Assignment 0 Part 2
# October 10, 2018
# Makefile

CC = g++
FLAGS = -g -Wall -std=c++11

# The following line is a relative directory reference that assumes the Eigen
# folder--which your program will depend on--is located one directory above the
# directory that contains this Makefile.
INCLUDE = -isystem../
SOURCES = *.cpp

EXENAME = transform_read

all: $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(INCLUDE) $(SOURCES)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
