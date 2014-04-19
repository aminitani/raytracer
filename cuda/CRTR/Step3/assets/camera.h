#pragma once

#include "../math/transform.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Camera
{
	private:
		float fovy;
		float aRatio;
		float vpd;//view plane distance, not significant, assume 1
		int m_width;
		int m_height;
	
	public:
		Transform orientation;

		// Camera()
		// {
			// orientation.Identify();
		// }
		
		Camera(Transform inTransform, float inFovy, int width, int height)
		{
			orientation = inTransform;
			fovy = inFovy;
			m_width = width;
			m_height = height;
			aRatio = (float)m_width/(float)m_height;
			vpd = 1;
		}

		Camera(const Camera &camera)
		{
			this->orientation = camera.orientation;
			this->vpd = camera.vpd;
			this->fovy = camera.fovy;
			this->aRatio = camera.aRatio;
			this->m_width = camera.m_width;
			this->m_height = camera.m_height;
		}
		
		CUDA_CALLABLE_MEMBER float Fovy() {return fovy;}
		CUDA_CALLABLE_MEMBER float &ARatio() {return aRatio;}
		CUDA_CALLABLE_MEMBER float VPD() {return vpd;}
		CUDA_CALLABLE_MEMBER int Width() {return m_width;}
		CUDA_CALLABLE_MEMBER int Height() {return m_height;}
};