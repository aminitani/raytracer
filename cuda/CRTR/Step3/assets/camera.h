#pragma once

#include "../math/transform.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct ViewPlane
{
	float vert, hor;
	Vec3 VPCenter, ULCorner;

	CUDA_CALLABLE_MEMBER ViewPlane() {}

	CUDA_CALLABLE_MEMBER ViewPlane(float &Afovy, float &Avpd, float &AaRatio, Transform &Aorientation)
	{
		//these are half the viewplane dimensions
		//TODO: tan is a bad function. works for most reasonable values though.
		vert = (float) tan(Afovy / 2.0) * Avpd;
		hor = vert * AaRatio;
		VPCenter = Aorientation.Pos() + ( Aorientation.Forward() * Avpd );
		ULCorner = VPCenter + Aorientation.Left() * hor + Aorientation.Up() * vert;
	}
};

struct Camera
{

	private:
		float fovy;
		float aRatio;
		float vpd;//view plane distance, not significant, assume 1
		int m_width;
		int m_height;
		float centerDistance/*, eye is orientation.Pos()*/;
	
	public:
		Transform orientation;
		ViewPlane viewPlane;

		// Camera()
		// {
			// orientation.Identify();
		// }
		
		CUDA_CALLABLE_MEMBER Camera(Transform inTransform, float inFovy, int width, int height)
		{
			orientation = inTransform;
			fovy = inFovy;
			m_width = width;
			m_height = height;
			aRatio = (float)m_width/(float)m_height;
			vpd = 1;
			viewPlane = ViewPlane(fovy, vpd, aRatio, orientation);
			centerDistance = /*(Vec3() - */orientation.Pos()/*)*/.length();
		}

		CUDA_CALLABLE_MEMBER Camera(const Camera &camera)
		{
			this->orientation = camera.orientation;
			this->vpd = camera.vpd;
			this->fovy = camera.fovy;
			this->aRatio = camera.aRatio;
			this->m_width = camera.m_width;
			this->m_height = camera.m_height;
			this->viewPlane = ViewPlane(fovy, vpd, aRatio, orientation);
			this->centerDistance = camera.centerDistance;
		}

		CUDA_CALLABLE_MEMBER ~Camera() {}

		CUDA_CALLABLE_MEMBER void Zoom(int distance)
		{
			orientation.Translate( orientation.Forward() * distance );
			centerDistance -= distance;
		}
		
		CUDA_CALLABLE_MEMBER float Fovy() {return fovy;}
		CUDA_CALLABLE_MEMBER float &ARatio() {return aRatio;}
		CUDA_CALLABLE_MEMBER float VPD() {return vpd;}
		CUDA_CALLABLE_MEMBER int Width() {return m_width;}
		CUDA_CALLABLE_MEMBER int Height() {return m_height;}
		CUDA_CALLABLE_MEMBER Vec3 Center() {return orientation.Pos() + orientation.Forward() * centerDistance;}
		CUDA_CALLABLE_MEMBER ViewPlane GetViewPlane() {return viewPlane;}
};