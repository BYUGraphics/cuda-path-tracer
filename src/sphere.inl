#pragma once
//sphere.inl
//11/25/15


namespace Sphere{
	
	struct Sphere{
		glm::vec3 position;
		float radius;
		int materialIdx;
	};
	
	//this function assumes that _rayDir is normalized. It also assumes that _intrsctNorm and _texCoord have been initialized
	//do a regular-old ray-sphere intersection test
	__device__ bool intersectSphere(Sphere *_sphere, glm::vec3 _rayOrig, glm::vec3 _rayDir, float *_intrsctDist, glm::vec3 *_intrsctNorm, glm::vec2 *_texCoord, int *_matIdx){
		//transform the ray into object space
		_rayOrig -= _sphere->position;
		
		//compute coefficients for the quadratic equation
		float a = glm::dot(_rayDir, _rayDir);
		float b = 2.f * glm::dot(_rayDir, _rayOrig);
		float c = glm::dot(_rayOrig, _rayOrig) - (_sphere->radius * _sphere->radius);
		
		//find the discriminant
		float disc = b * b - 4.f * a * c;
		
		//if there are no real roots, there's no intersection
		if(disc < 0.f){
			//TODO: indicate that there was no intersection
			return false;
		}
		
		//compute q
		float discSqrt = glm::sqrt(disc);
		float q;
		if(b < 0.f){
			q = -0.5f * (b - discSqrt);
		}
		else{
			q = -0.5f * (b + discSqrt);
		}
		
		//compute intersection distances
		float t0 = q / a;
		float t1;
		if(q != 0.f){
			t1 = c / q;
		}
		else{
			t1 = 1e30f;	//this ensures that t0 is closer to the ray origin
		}
		
		//get the closer point
		if(t1 < t0){
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		
		//store the intersection distance, normal, and material index
		*(_intrsctDist) = t0;
		*(_matIdx) = _sphere->materialIdx;
		glm::vec3 pt = _rayOrig + t0 * _rayDir;
		glm::vec3 norm = glm::normalize(pt - _sphere->position);
		_intrsctNorm->x = norm.x;
		_intrsctNorm->y = norm.y;
		_intrsctNorm->z = norm.z;
		
		//Get the sphere's texture coordinate
		_texCoord->x = 0.f;
		_texCoord->y = 0.f;
		//TODO
		
		return true;
	}
	
}




