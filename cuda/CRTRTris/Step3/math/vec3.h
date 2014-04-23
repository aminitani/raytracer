#pragma once

#include <iostream>
#include <cmath>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

const double GR_PI = 3.1415926535897932384626433832795;

struct Vec3
{
public:
	float x, y, z;
	CUDA_CALLABLE_MEMBER Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	CUDA_CALLABLE_MEMBER Vec3(float xx) : x(xx), y(xx), z(xx) {}
	CUDA_CALLABLE_MEMBER Vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
	CUDA_CALLABLE_MEMBER Vec3& normalize()
	{
		float nor2 = length2();
		if (nor2 > 0) {
			float invNor = 1 / sqrt(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}
	CUDA_CALLABLE_MEMBER Vec3 operator * (const float &f) const { return Vec3(x * f, y * f, z * f); }
	CUDA_CALLABLE_MEMBER Vec3 operator * (const Vec3 &v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
	CUDA_CALLABLE_MEMBER float dot(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
	CUDA_CALLABLE_MEMBER Vec3 cross(const Vec3 &v) const { return Vec3( (y*v.z - z*v.y) , (z*v.x - x*v.z), (x*v.y - y*v.x) ); }
	CUDA_CALLABLE_MEMBER Vec3 operator - (const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
	CUDA_CALLABLE_MEMBER Vec3 operator + (const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
	CUDA_CALLABLE_MEMBER Vec3& operator += (const Vec3 &v) { x += v.x, y += v.y, z += v.z; return *this; }
	CUDA_CALLABLE_MEMBER Vec3& operator *= (const Vec3 &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
	CUDA_CALLABLE_MEMBER Vec3 operator - () const { return Vec3(-x, -y, -z); }
	CUDA_CALLABLE_MEMBER float length2() const { return x * x + y * y + z * z; }
	CUDA_CALLABLE_MEMBER float length() const { return sqrt(length2()); }
	CUDA_CALLABLE_MEMBER static float Distance(Vec3 a, Vec3 b) { return sqrt( (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y)); }
	friend std::ostream & operator << (std::ostream &os, const Vec3 &v)
	{
		os << "(" << v.x << " " << v.y << " " << v.z << ")";
		return os;
	}
};