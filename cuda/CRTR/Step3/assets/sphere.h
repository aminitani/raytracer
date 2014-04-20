#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

class Sphere {
public:
	Vec3 center;
	float radius;
	float radius2;

	CUDA_CALLABLE_MEMBER Sphere(Vec3 c, float r, Vec3 _color) : color(_color), center(c), radius(r), radius2(r*r) {}

	CUDA_CALLABLE_MEMBER Sphere(const Sphere &sphere)
	{
		this->center = sphere.center;
		this->radius = sphere.radius;
		this->radius2 = sphere.radius2;
	}

	CUDA_CALLABLE_MEMBER bool Intersect(Ray &ray, float *t0 = NULL)
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

	CUDA_CALLABLE_MEMBER Vec3 Normal(Vec3 phit) {
		Vec3 normal = phit - center;
		normal.normalize();
		return normal;
	}

	CUDA_CALLABLE_MEMBER Vec3 Color() {return color;}

protected:
	Vec3 color;
};