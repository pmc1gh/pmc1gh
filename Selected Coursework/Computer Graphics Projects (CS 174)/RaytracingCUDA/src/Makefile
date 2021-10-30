# CS 179 Project: GPU-accelerated ray tracing
# Team members: Philip Carr and Thomas Leing
# June 10, 2019
# Makefile

CC = /usr/bin/g++

LD_FLAGS = -lrt -lGLEW -lGL -lGLU -lglut -lm -lpng# -g -ggdb3
#LD_FLAGS = -lrt -lGLEW -lGL -lGLU -lglut -lm -lpng -g -fsanitize=address -ggdb3

CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_61,code=sm_61 \
        -gencode arch=compute_61,code=compute_61

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname),Darwin)
	LDFLAGS       := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
	CCFLAGS   	  := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS   := -m32
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := -L$(CUDA_LIB_PATH) -lcudart
		CCFLAGS       := -m64
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32
else
	NVCCFLAGS := -m64
endif

TARGETS = raytrace

all: $(TARGETS)

raytrace: *.cpp raytrace_device.o
	#$(CC) $^ -o $@ -O3 $(LDFLAGS) -lrt -lGLEW -lGL -lGLU -lglut -lm -lpng -Wall -I$(CUDA_INC_PATH)
	# this works: $(CC) $^ -o $@ -O3 $(LDFLAGS) -Wall -I$(CUDA_INC_PATH)
	# these don't: $(CC) $^ -o $@ -O3 $(LDFLAGS) -lrt -lGLEW -lGL -lGLU -lglut -lm -lpng -g -fsanitize=address -ggdb3 -Wall -I$(CUDA_INC_PATH)
	#$(CC) $^ -o $@ -O0 $(LDFLAGS) $(LD_FLAGS) -Wall -I$(CUDA_INC_PATH)
	$(CC) $^ -o $@ -O0 $(LDFLAGS) $(LD_FLAGS) -w -I$(CUDA_INC_PATH)

raytrace_device.o: raytrace_device.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

bunny: $(TARGETS)
	./raytrace scene_bunny.txt 150 150 1

armadillo: $(TARGETS)
	./raytrace scene_armadillo.txt 150 150 1

test: $(TARGETS)
	./raytrace scene_bunny.txt 150 150 1
	mv rt.png rt_test_gpu_bunny_150x150.png
	./raytrace scene_armadillo.txt 150 150 1
	mv rt.png rt_test_gpu_armadillo_150x150.png

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
