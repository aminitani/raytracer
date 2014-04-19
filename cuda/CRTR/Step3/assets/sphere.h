#pragma once
#include "./object.h"

class Sphere : public Object {
public:
	Vec3 center;
	float radius;
	float radius2;
	Sphere(Vec3 c, float r, Vec3 _color) : Object(_color), center(c), radius(r), radius2(r*r) {}

	virtual bool Intersect(Ray &ray, float *t0 = NULL)
	{
		Vec3 l = center - ray.Point();
		float tca = l.dot(ray.Direction());
		if (tca < 0) return false;
		float d2 = l.dot(l) - tca * tca;
		if (d2 > radius2) return false;
		float thc = sqrt(radius2 - d2);
		if (t0 != NULL) {
			*t0 = tca - thc;
			if(*t0 < 0)
				*t0 = tca + thc;
		}

		return true;
	}

	virtual Vec3 Normal(Vec3 phit) {
		Vec3 normal = phit - center;
		normal.normalize();
		return normal;
	}
};