INCLUDE_PATHS = -I lib/ -I src/

build:
	nvcc src/*.cu src/*.cpp $(INCLUDE_PATHS) -o bin/pathtrace