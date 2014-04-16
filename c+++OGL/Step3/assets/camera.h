#pragma once

#include "../math/transform.h"

class Camera
{
	private:
		float fovy;
		float aRatio;
		float vpd;//view plane distance, not significant, assume 1
	
	public:
		Transform orientation;

		// Camera()
		// {
			// orientation.Identify();
		// }
		
		Camera(Transform inTransform, float inFovy, float inARatio)
		{
			orientation = inTransform;
			fovy = inFovy;
			aRatio = inARatio;
			vpd = 1;
		}
		
		float Fovy() {return fovy;}
		float &ARatio() {return aRatio;}
		float VPD() {return vpd;}
};