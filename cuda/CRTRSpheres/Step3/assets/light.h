#pragma once
#include "../math/vec3.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Light
{
	private:
		//TODO: replace brightness with color
		float brightness;
	public:
		Vec3 position;
		
		CUDA_CALLABLE_MEMBER Light(Vec3 pos, float bright)
		{
			position = pos;
			brightness = bright;
		}

		CUDA_CALLABLE_MEMBER float Brightness(float distance)
		{
			return brightness;// / (distance * distance);
		}
};