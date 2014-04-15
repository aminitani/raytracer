//TODO: multiple additive lights
//TODO: inverse square light falloff

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include "./math/transform.h"
#include "./assets/image.h"
#include "./assets/camera.h"
#include "./assets/sphere.h"
#include "./assets/triangle.h"
#include "./assets/light.h"
#include "./math/ray.h"

using std::cout; using std::endl;
using std::vector;

#define CollisionError 0.05

class Raytracer
{
	private:
		Image *image;
		Camera *camera;
		vector<Object *> objects;
		// Light * light;
		vector<Light *> lights;
		
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
			lights.push_back(new Light(Vec3(3.0,3.0,3.0), 0.8));
			// light = new Light(Vec3(-3.0,3.0,3.0), 0.8);
			Sphere *sphere = new Sphere(Vec3(0.0,0.0,-5.0), 3.0, Vec3(0.7,0.0,0.0));
			objects.push_back(sphere);
			objects.push_back(new Triangle(
				Vec3(-2.0, 0.0, 1.0),
				Vec3( 2.0, 0.0, 1.0),
				Vec3( 0.0, 3.0, 1.0),
				Vec3(0.0,0.7,0.0)));
		}
	
	public:
		Raytracer()
		{
			image = new Image(640, 480, "test.png");
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
			//test
			std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
			
			//this is the color to be contributed to the pixels
			Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);
			
			//iterate from upper left to lower right through all pixels
			for (unsigned i = 0; i < image->Width(); ++i)
			{
				for (unsigned j = 0; j < image->Height(); ++j)
				{
					
					// compute primary ray direction
					Ray primaryRay;
					ComputePrimaryRay(i, j, &primaryRay);
					// cout << "primary ray point: " << primaryRay.Point() << " Direction: " << primaryRay.Direction() << endl;
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
						Vec3 color(0);

						// compute illumination
						Ray shadowRay;
						for(auto * light : lights) {
							shadowRay.Point() = phit;
							shadowRay.Direction() = (light->position - phit/*(pHit + CollisionError * nhit)*/).normalize();
							bool isInShadow = false;
							for (unsigned k = 0; k < objects.size(); ++k) {
			//					if (Intersect(objects[k], shadowRay)) {
								float t0 = std::numeric_limits<float>::infinity();
								if(objects[k] != object)
								if ((*objects[k]).Intersect(shadowRay, &t0)) {
									isInShadow = true;
									break;
								}
							}
							if (!isInShadow) {
								color += (object->Color() * std::max(0.0f, nhit.dot(shadowRay.Direction())) * light->brightness);
							}
							// else
							// 	pixel.SetColor(Vec3(0.0));
						}
						pixel.SetColor(color);
					}
					else
						pixel.SetColor(Vec3(0.3));
					image->Contribute(i, j, pixel);
				}
			}

			//test
			auto duration = std::chrono::high_resolution_clock::now() - start;
			cout << "Time: " << (float)duration.count() / 1000000 << " milliseconds." << endl << endl;
			
			image->Save();
		}
};