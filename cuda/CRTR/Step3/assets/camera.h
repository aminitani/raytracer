#pragma once

#include "../math/transform.h"

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
		
		float Fovy() {return fovy;}
		float &ARatio() {return aRatio;}
		float VPD() {return vpd;}
		int Width() {return m_width;}
		int Height() {return m_height;}
};