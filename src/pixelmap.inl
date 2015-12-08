/*
*	Pixelmap namespace
*	Author(s): Jeremy Oborn
*	Date: 12/07/2015
*
*	This namespace loads a .ppm file and stores it in a pixelmap struct that makes it easy to pass all of the data to the GPU.
*
*/
#pragma once
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace Pixelmap
{
	struct Pixelmap
	{
		int width, height, total_size;
		glm::vec3 *pixels;

	};

}

