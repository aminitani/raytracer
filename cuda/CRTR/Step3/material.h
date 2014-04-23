#include "math\vec3.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Material
{
	Vec3 color;
	float IOR;
	float transparency, reflection;

	CUDA_CALLABLE_MEMBER Material(float inTransparency = 0, float inReflection = 0, float inIOR = 1.5, Vec3 inColor = Vec3(.5))
	{
		color = inColor;
		IOR = inIOR;
		transparency = inTransparency;
		reflection = inReflection;
	}
};