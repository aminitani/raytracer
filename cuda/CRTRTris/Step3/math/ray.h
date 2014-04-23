#pragma once
#include "vec3.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Ray
{
	private:
		Vec3 point;
		Vec3 dir;
	public:
		CUDA_CALLABLE_MEMBER Ray()
		{
			point = Vec3();
			dir = Vec3();
		}
		
		CUDA_CALLABLE_MEMBER Ray(Vec3 inPoint, Vec3 inDir)
		{
			point = inPoint;
			dir = inDir;
		}
		
		CUDA_CALLABLE_MEMBER Vec3 &Point() { return point; }
		
		CUDA_CALLABLE_MEMBER Vec3 &Direction() { return dir; }
};