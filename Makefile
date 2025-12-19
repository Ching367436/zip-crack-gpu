NVFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp

# Headers from minizip-ng and its compat shim providing unzip.h
MINIZIPFLAGS := -I./deps/minizip-ng 
# Library search path and libs produced by minizip-ng build
MINIZIP_LIBDIR := ./deps/minizip-ng/build
MINIZIP_CORE := $(wildcard $(MINIZIP_LIBDIR)/libmz.a)
MINIZIP_LIBS := $(MINIZIP_LIBDIR)/libminizip.a $(MINIZIP_CORE) -lz

TARGET := final_project

.PHONY: all deps clean

all: deps $(TARGET)

# Build minizip-ng in deps (once). Adjust generator if needed.
deps:
	cmake -S ./deps/minizip-ng -B $(MINIZIP_LIBDIR) \
	  -DMZ_BUILD_TEST=OFF \
	  -DMZ_ZLIB=ON \
	  -DMZ_ZSTD=OFF \
	  -DMZ_BZIP2=OFF \
	  -DMZ_LZMA=OFF \
	  -DMZ_LIBCOMP=OFF \
	  -DMZ_OPENSSL=OFF \
	  -D MZ_PKCRYPT=ON \
	  -D MZ_WZAES=OFF \
      -DMZ_COMPAT=ON \
      -DMZ_BUILD_SHARED=OFF \
      -DMZ_BUILD_STATIC=ON
	cmake --build $(MINIZIP_LIBDIR)

kernel.o: kernel.cu
	nvcc $(NVFLAGS) -c kernel.cu -o kernel.o

$(TARGET): final_project.cu kernel.o
	nvcc $(NVFLAGS) $(MINIZIPFLAGS) -o $(TARGET) final_project.cu kernel.o $(MINIZIP_LIBS)

clean:
	rm -rf $(TARGET) *.o

