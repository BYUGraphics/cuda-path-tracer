#pragma once
//sphere.inl
//11/25/15


namespace Sphere{
	
	struct Sphere{
		glm::vec3 position;
		float radius;
		int materialIdx;
	};
	
	__device__ void intersectSphere(Sphere *_sphere, glm::vec3 _rayPos, glm::vec3 _rayDir, float &_intrsctDist, glm::vec3 &_intrsctNorm, glm::vec2 &_texCoord, int &_matIdx){
		//do a regular-old ray-sphere intersection test
	}
	
}
