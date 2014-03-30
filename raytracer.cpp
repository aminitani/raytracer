#include <iostream>
#include <vector>
#include "./RayMath.h"
#include "./math/transform.h"
#include "./assets/image.h"
#include "./assets/camera.h"

using std::cout; using std::endl;
using std::vector;

int main()
{
	// RayMath::print();
	
	Image image(1280, 720, "test.png");
	//camera is placed at (0,0,-5) and faces in the negative z direction, looking at origin
	float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,-5.0,1.0};
	Camera camera(Transform(camTrans), (float)55.0, (float)image.Width()/(float)image.Height());
	
	//trace some rays///////////////////////////////////////
	
	for(unsigned y = 0; y < image.Height(); y++)
	for(unsigned x = 0; x < image.Width(); x++)
	{
		Pixel pixel( (int)( (float)x/(float)image.Width()*255.0f ),
			0,
			(int)( (float)y/(float)image.Height()*255.0f ),
			(int)( sqrt( (float)x/(float)image.Width()*(float)y/(float)image.Height() )*255.0f ) );
		image.Contribute(x, y, pixel);
	}
	
	//trace some rays///////////////////////////////////////
	
	image.Save();
	
	return 0;
}
