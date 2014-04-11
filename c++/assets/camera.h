#include "../math/transform.h"

class Camera
{
	private:
		Transform orientation;
		float fovy = 0;
		float aRatio = 0;
		float vpd = 1;//view plane distance, not significant, assume 1
	
	public:
		// Camera()
		// {
			// orientation.Identify();
		// }
		
		Camera(Transform inTransform, float inFovy, float inARatio)
		{
			orientation = inTransform;
			fovy = inFovy;
			aRatio = inARatio;
		}
		
		float Fovy() {return fovy;}
		float ARatio() {return aRatio;}
		float VPD() {return vpd;}
		Transform Orientation() {return orientation;}
};