#pragma once
#include "../math/vec3.h"
#include "../math/ray.h"

class Object {
public:
	Object(Vec3 c) : color(c) {}
	virtual bool Intersect(Ray, float*) = 0;
	virtual Vec3 Normal(Vec3) = 0;
	Vec3 Color() {return color;}
protected:
	Vec3 color;
};