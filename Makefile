INCLUDE_PATHS = -I lib/ -I src/

build:
	nvcc src/*.cu $(INCLUDE_PATHS) -o bin/pathtrace
	