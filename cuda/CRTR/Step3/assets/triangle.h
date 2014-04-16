#include "./object.h"

class Triangle : public Object {
public:
	Vec3 vert[3];
	Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 color) : Object(color) {
		vert[0] = v1;
		vert[1] = v2;
		vert[2] = v3;
	}

	virtual bool Intersect(Ray &ray, float* t0) {
		Vec3 tvec = ray.Point() - vert[0];  
		Vec3 pvec = ray.Direction().cross(vert[2]);  
		float  det  = vert[1].dot(pvec);

		det = 1.0f / det;

		float u = tvec.dot(pvec) * det;  

		if (u < 0.0f || u > 1.0f)
			return false;  

		Vec3 qvec = tvec.cross(vert[1]);  

		float v = ray.Direction().dot(qvec) * det;  

		if (v < 0.0f || (u + v) > 1.0f)  
			return false;

		if(t0 != NULL)
			*t0 = vert[2].dot(qvec) * det;
		return true;
	}

	virtual Vec3 Normal(Vec3 phit) {
		Vec3 normal = vert[1].cross(vert[2]);
		normal.normalize();
		return normal;
	}
};