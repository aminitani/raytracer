#include "./object.h"

class Triangle : public Object {
public:
	Vec3 vert[3];
	Triangle(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 color) : Object(color) {
		vert[0] = v1;
		vert[1] = v2;
		vert[2] = v3;
	}

	/*virtual bool Intersect(Ray ray, float* t0) {
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
	}*/

	virtual bool Intersect(Ray ray, float * t0) {
		Vec3 v0v1 = vert[1] - vert[0];
        Vec3 v0v2 = vert[2] - vert[0];
        Vec3 N = v0v1.cross(v0v2);
        float nDotRay = N.dot(ray.Direction());
        if (nDotRay == 0) return false; // ray parallel to triangle 
        float d = N.dot(vert[0]);
        float t = -(N.dot(ray.Point()) + d) / nDotRay;

        // inside-out test
        Vec3 Phit = ray.Point() + ray.Direction() * t;
  
       // inside-out test edge0
        Vec3 v0p = Phit - vert[0];
        float v = N.dot(v0v1.cross(v0p));
        if (v < 0) return false; // P outside triangle
 
        // inside-out test edge1
        Vec3 v1p = Phit - vert[1];
        Vec3 v1v2 = vert[2] - vert[1];
        float w = N.dot(v1v2.cross(v1p));
        if (w < 0) return false; // P outside triangle
 
        // inside-out test edge2
        Vec3 v2p = Phit - vert[2];
        Vec3 v2v0 = vert[0] - vert[2];
        float u = N.dot(v2v0.cross(v2p));
        if (u < 0) return false; // P outside triangle
 
 		if(t0 != NULL)
 			*t0 = t;

 		return true;
	}

	virtual Vec3 Normal(Vec3 phit) {
		Vec3 edge1 = vert[1] - vert[0];
		Vec3 edge2 = vert[2] - vert[0];
		Vec3 normal = edge1.cross(edge2);
		normal.normalize();
		return normal;
	}
};