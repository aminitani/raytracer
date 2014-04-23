#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define EPSILON 0.000000000008854

class Triangle {
public:
	Vec3 v0, v1, v2;

	CUDA_CALLABLE_MEMBER Triangle(Vec3 vert0, Vec3 vert1, Vec3 vert2, Vec3 _normal, Vec3 _color) : v0(vert0), v1(vert1), v2(vert2), color(_color), normal(_normal) {}

	CUDA_CALLABLE_MEMBER Triangle(const Triangle &triangle) {
		this->v0 = triangle.v0;
		this->v1 = triangle.v1;
		this->v2 = triangle.v2;
		this->normal = triangle.normal;
		this->color = triangle.color;
	}

	CUDA_CALLABLE_MEMBER bool Intersect(Ray &ray, float* t0) {
		Vec3 edge1 = v1 - v0; 
		Vec3 edge2 = v2 - v0; 
		Vec3 pvec = ray.Direction().cross(edge2); 
		float det = edge1.dot(pvec); 

		if (det > -EPSILON && det < EPSILON) 
			return false; 

		float invDet = 1 / det; 
		Vec3 tvec = ray.Point() - v0; 
		float u = tvec.dot(pvec) * invDet;

		if (u < 0 || u > 1) 
			return false; 

		Vec3 qvec = tvec.cross(edge1); 
		float v = ray.Direction().dot(qvec) * invDet; 

		if (v < 0 || u + v > 1) 
			return false; 

		if(t0 != NULL)
			*t0 = edge2.dot(qvec) * invDet; 

		return true; 
	}

	CUDA_CALLABLE_MEMBER Vec3 Normal() {
		/*
		Vec3 edge1 = v1 - v0;
		Vec3 edge2 = v2 - v0;
		Vec3 normal = edge1.cross(edge2);
		normal.normalize();
		*/
		return normal;
	}

	CUDA_CALLABLE_MEMBER Vec3 Color() {return color;}

protected:
	Vec3 normal; //average of vector normals?
	Vec3 color;
};