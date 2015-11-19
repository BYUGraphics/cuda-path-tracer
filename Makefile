INCLUDE_PATH = -I lib/ -I src/

build:
	nvcc src/*.cu $(INCLUDE_PATH) -o bin/pathtrace