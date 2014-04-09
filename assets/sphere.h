#include "../math/vec3.h"

class Sphere {
private:
	Vec3 center;
	Vec3 color;
	float radius;
public:
	Sphere();
};

Sphere::Sphere(Vec3 _center, Vec3 _color, float r) : center(c), color(_color), radius(r) {
}