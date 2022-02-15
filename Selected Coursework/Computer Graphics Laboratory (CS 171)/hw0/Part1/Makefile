# Philip Carr
# CS 171 Assignment 0 Part 1
# October 10, 2018
# Makefile

CC = g++
FLAGS = -g -Wall -std=c++11
SOURCES = *.h *.cpp

EXENAME = object_read

all: $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(SOURCES)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
