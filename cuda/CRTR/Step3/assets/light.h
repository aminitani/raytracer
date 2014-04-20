#pragma once
#include "../math/vec3.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Light
{
	public:
		Vec3 position;
		//TODO: replace brightness with color
		float brightness;
		
		Light(Vec3 pos, float bright)
		{
			position = pos;
			brightness = bright;
		}
};