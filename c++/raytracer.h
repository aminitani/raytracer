#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "./RayMath.h"
#include "./math/transform.h"
#include "./assets/image.h"
#include "./assets/camera.h"
#include "./assets/sphere.h"
#include "./assets/triangle.h"
#include "./math/ray.h"

using std::cout; using std::endl;
using std::vector;

class Raytracer
{
	private:
		Image *image;
		Camera *camera;
		vector<Object *> objects;
		Vec3 lightPosition;
		
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

		void ConstructScene() {
			lightPosition = Vec3(3.0,3.0,3.0);
			Sphere *sphere = new Sphere(Vec3(0.0,0.0,0.0), 5.0, Vec3(0.5,0.0,0.0));
			objects.push_back(sphere);
			// objects.push_back(Sphere(Vec3(0.0,0.0,0.0), 5.0, Vec3(0.5,0.0,0.0)));
		}
	
	public:
		Raytracer()
		{
			image = new Image(1280, 720, "test.png");
			//camera is placed at (0,0,5) and faces in the negative z direction, looking at origin
			float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,5.0,1.0};
			camera = new Camera(Transform(camTrans), 3.1415926 * 55.0 / 180.0, (float)image->Width()/(float)image->Height());
			ConstructScene();
		}
		
		~Raytracer()
		{
			delete image;
			image = NULL;
		}
		
		void Start()
		{
			//iterate from upper left to lower right through all pixels
			for (unsigned i = 0; i < image->Width(); ++i)
			{
				for (unsigned j = 0; j < image->Height(); ++j)
				{
					//this is the color to be contributed to this pixel
					Pixel pixel( (int)( (float)i/(float)image->Width()*255.0f ),
						0,
						(int)( (float)j/(float)image->Height()*255.0f ),
						(int)( sqrt( (float)i/(float)image->Width()*(float)j/(float)image->Height() )*255.0f ) );
						
					// compute primary ray direction
					Ray primaryRay;
					ComputePrimaryRay(i, j, &primaryRay);
					cout << "primary ray point: " << primaryRay.Point() << " Direction: " << primaryRay.Direction() << endl;
					// shoot prim ray in the scene and search for intersection
					Point pHit;
					Normal nHit;
					float minDist = std::numeric_limits<float>::infinity();
					Object *object = NULL;
					for (unsigned k = 0; k < objects.size(); ++k) {
		//				if (Intersect(objects[k], primRay, &pHit, &nHit)) {
						float t0 = std::numeric_limits<float>::infinity();
						if ((*objects[k]).Intersect(primaryRay, &t0)) {
							// float distance = Vec3::Distance(eyePosition, pHit);
							if (t0 < minDist) {
								object = objects[k];
								minDist = t0; // update min distance
							}
						}
					}
					if (object != NULL) {
						// compute phit and nhit
						Vec3 phit = primaryRay.Point() + primaryRay.Direction() * minDist; // point of intersection
						Vec3 nhit = object->Normal(phit);

						// compute illumination
						Ray shadowRay;
						shadowRay.Direction() = lightPosition - pHit;
						bool isInShadow = false;
						for (unsigned k = 0; k < objects.size(); ++k) {
		//					if (Intersect(objects[k], shadowRay)) {
							float t0 = std::numeric_limits<float>::infinity();
							if ((*objects[k]).Intersect(shadowRay, &t0)) {
								isInShadow = true;
								break;
							}
						}
						// if (!isInShadow)
			//				pixel.SetColor(object->color * light.brightness);
			//			else
			//				pixel.SetColor(Vec3(0.0));
					}
					image->Contribute(i, j, pixel);
				}
			}

			//sample image generation
			// for(unsigned y = 0; y < image->Height(); y++)
			// {
				// for(unsigned x = 0; x < image->Width(); x++)
				// {
					// Pixel pixel( (int)( (float)x/(float)image->Width()*255.0f ),
						// 0,
						// (int)( (float)y/(float)image->Height()*255.0f ),
						// (int)( sqrt( (float)x/(float)image->Width()*(float)y/(float)image->Height() )*255.0f ) );
					// image->Contribute(x, y, pixel);
				// }
			// }
			
			image->Save();
		}
};