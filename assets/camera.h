#include "../math/transform.h"

class Camera
{
	private:
		Transform transform;
		float xFov = 0;
		float aRatio = 0;
	
	public:
		// Camera()
		// {
			// transform.Identify();
		// }
		
		Camera(Transform inTransform, float inXFov, float inARatio)
		{
			transform = inTransform;
			xFov = inXFov;
			aRatio = inARatio;
		}
		
		float XFov() {return xFov;}
		float ARatio() {return aRatio;}
};