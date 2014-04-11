#pragma once
#include "../math/vec3.h"
#include "../math/ray.h"

class Object {
public:
	Vec3 color;
	virtual bool Intersect(Ray, float*) = 0;
	virtual Vec3 Normal(Vec3) = 0;
protected:
	Object(Vec3 c) : color(c) {}
};