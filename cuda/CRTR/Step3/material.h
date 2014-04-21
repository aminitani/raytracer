#include "math\vec3.h"

struct Material
{
	Vec3 color;
	float IOR;
	bool isTransmissive;

	Material(bool transmissive = false, float inIOR = 1.5, Vec3 inColor = Vec3(.5))
	{
		color = inColor;
		IOR = inIOR;
		isTransmissive = transmissive;
	}
};