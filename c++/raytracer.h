#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "./RayMath.h"
#include "./math/transform.h"
#include "./assets/image.h"
#include "./assets/camera.h"
#include "./math/ray.h"

using std::cout; using std::endl;
using std::vector;

typedef Vec3 Point;
typedef Vec3 Normal;

class Raytracer
{
	private:
		Image *image;
		Camera *camera;
		
		void ComputePrimaryRay(unsigned x, unsigned y, Ray *primaryRay)
		{
			primaryRay->Point() = camera->Orientation().Pos();
			//these are half the viewplane dimensions
			//TODO: tan is a bad function. works for most reasonable values though.
			float vert = tan(camera->Fovy() / 2.0) * camera->VPD();
			float hor = vert * camera->ARatio();
			Point VPCenter = camera->Orientation().Pos() + ( camera->Orientation().Forward() * camera->VPD() );
			Point ULCorner = VPCenter + camera->Orientation().Left() * hor + camera->Orientation().Up() * vert;
			// cout << "cameye: " << camera->Orientation().Pos() << " VPC: " << VPCenter << " ULC: " << ULCorner << endl;
			Point target =
			ULCorner - camera->Orientation().Left() * ( ((x+.5) / image->Width()) * 2*hor )
			- camera->Orientation().Up() * ( ((y+.5) / image->Height()) * 2*vert );
			primaryRay->Direction() = (target - camera->Orientation().Pos()).normalize();
		}
	
	public:
		Raytracer()
		{
			image = new Image(1280, 720, "test.png");
			//camera is placed at (0,0,5) and faces in the negative z direction, looking at origin
			float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,5.0,1.0};
			camera = new Camera(Transform(camTrans), 3.1415926 * 55.0 / 180.0, (float)image->Width()/(float)image->Height());
		}
		
		~Raytracer()
		{
			delete image;
			image = NULL;
		}
		
		void Start()
		{
			//iterate from upper left to lower right through all pixels
			for (unsigned i = 0; i < image->Height(); ++i)
			{
				for (unsigned j = 0; j < image->Width(); ++j)
				{
					// compute primary ray direction
					Ray primaryRay;
					ComputePrimaryRay(i, j, &primaryRay);
					cout << "primary ray point: " << primaryRay.Point() << " Direction: " << primaryRay.Direction() << endl;
					// shoot prim ray in the scene and search for intersection
					Point pHit;
					Normal nHit;
					float minDist = std::numeric_limits<float>::infinity();
		//			Object object = NULL;
		//			for (int k = 0; k < objects.size(); ++k) {
		//				if (Intersect(objects[k], primRay, &pHit, &nHit)) {
		//					float distance = Distance(eyePosition, pHit);
		//					if (distance < minDistance) {
		//						object = objects[k];
		//						minDistance = distance; // update min distance
		//					}
		//				}
		//			}
		//			if (object != NULL) {
						// compute illumination
		//				Ray shadowRay;
		//				shadowRay.direction = lightPosition - pHit;
		//				bool isShadow = false;
		//				for (int k = 0; k < objects.size(); ++k) {
		//					if (Intersect(objects[k], shadowRay)) {
		//						isInShadow = true;
		//						break;
		//					}
		//				}
		//			}
		//			if (!isInShadow)
		//				pixels[i][j] = object->color * light.brightness;
		//			else
		//				pixels[i][j] = 0;
				}
			}

			//sample image generation
			for(unsigned y = 0; y < image->Height(); y++)
			{
				for(unsigned x = 0; x < image->Width(); x++)
				{
					Pixel pixel( (int)( (float)x/(float)image->Width()*255.0f ),
						0,
						(int)( (float)y/(float)image->Height()*255.0f ),
						(int)( sqrt( (float)x/(float)image->Width()*(float)y/(float)image->Height() )*255.0f ) );
					image->Contribute(x, y, pixel);
				}
			}
			
			image->Save();
		}
};