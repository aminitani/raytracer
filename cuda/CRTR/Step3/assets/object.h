#pragma once
#include "../math/vec3.h"
#include "../math/ray.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Object {
public:
	Object(Vec3 c) : color(c) {}
	virtual bool Intersect(Ray &, float*) = 0;
	virtual Vec3 Normal(Vec3) = 0;
	Vec3 Color() {return color;}
protected:
	Vec3 color;
};