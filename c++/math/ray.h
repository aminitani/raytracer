#pragma once
#include "vec3.h"

class Ray
{
	private:
		Vec3 point;
		Vec3 dir;
	public:
		Ray()
		{
			point = Vec3();
			dir = Vec3();
		}
		
		Ray(Vec3 inPoint, Vec3 inDir)
		{
			point = inPoint;
			dir = inDir;
		}
		
		Vec3 &Point() { return point; }
		
		Vec3 &Direction() { return dir; }
};