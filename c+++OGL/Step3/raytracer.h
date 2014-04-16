//TODO: multiple additive lights
//TODO: inverse square light falloff

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <thread>
#include "./math/transform.h"
#include "./assets/image.h"
#include "./assets/camera.h"
#include "./assets/sphere.h"
#include "./assets/triangle.h"
#include "./assets/light.h"
#include "./math/ray.h"

using std::cout; using std::endl;
using std::vector;
using std::thread;

#define CollisionError 0.05

class Raytracer
{
	private:
		Image *image;
		Camera *camera;
		vector<Object *> objects;
		Light *light;
		float *pixels;//the pixel buffer shared between the childview and this raytracer
		float *localPixels;//write to this, then push into pixels when the whole image is done
		
		void ComputePrimaryRay(unsigned x, unsigned y, unsigned width, unsigned height, Ray *primaryRay)
		{
			primaryRay->Point() = camera->orientation.Pos();
			//these are half the viewplane dimensions
			//TODO: tan is a bad function. works for most reasonable values though.
			float vert = (float) tan(camera->Fovy() / 2.0) * camera->VPD();
			float hor = vert * camera->ARatio();
			Point VPCenter = camera->orientation.Pos() + ( camera->orientation.Forward() * camera->VPD() );
			Point ULCorner = VPCenter + camera->orientation.Left() * hor + camera->orientation.Up() * vert;
			// cout << "cameye: " << camera->orientation.Pos() << " VPC: " << VPCenter << " ULC: " << ULCorner << endl;
			Point target =
			ULCorner - camera->orientation.Left() * ( ((x+.5) / width) * 2*hor )
			- camera->orientation.Up() * ( ((y+.5) / height) * 2*vert );
			primaryRay->Direction() = (target - camera->orientation.Pos()).normalize();
		}

		void ConstructScene() {
			light = new Light(Vec3(-3.0,3.0,3.0), 0.8);
			Sphere *sphere = new Sphere(Vec3(1.0,1.0,-1.0), 3.0, Vec3(0.5,0.0,0.0));
			objects.push_back(sphere);
		}
	
	public:
		Raytracer(int width, int height, float *inPixels)
		{
			image = new Image(width, height, "test.png");
			//camera is placed at (0,0,5) and faces in the negative z direction, looking at origin
			float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,5.0,1.0};
			camera = new Camera(Transform(camTrans), 3.1415926 * 55.0 / 180.0, (float)width/(float)height/*(float)image->Width()/(float)image->Height()*/);
			pixels = inPixels;
			localPixels = new float[width*height*4];
			ConstructScene();
		}
		
		~Raytracer()
		{
			delete image;
			image = NULL;
			
			delete camera;
			camera = NULL;
			
			for(Object *obj : objects)
			{
				delete obj;
				obj = NULL;
			}
			
			delete light;
			light = NULL;
		}

		Camera *GetCamera() {return camera;}

		void Thrender(unsigned start, unsigned end)
		{
			for (unsigned i = start; i < end; ++i)
			{
				for (unsigned j = 0; j < image->Height(); ++j)
				{
					Trace(i, j);
				}
			}
		}

		void Trace(int i, int j)
		{
			//this is the color to be contributed to the pixels
			Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);
			// compute primary ray direction
			Ray primaryRay;
			ComputePrimaryRay(i, j, image->Width(), image->Height(), &primaryRay);
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

				// compute illumination
				Ray shadowRay;
				shadowRay.Point() = phit;
				shadowRay.Direction() = (light->position - phit/*(pHit + CollisionError * nhit)*/).normalize();
				bool isInShadow = false;
				for (unsigned k = 0; k < objects.size(); ++k) {
//					if (Intersect(objects[k], shadowRay)) {
					float t0 = std::numeric_limits<float>::infinity();
					if ((*objects[k]).Intersect(shadowRay, &t0)) {
						isInShadow = true;
						break;
					}
				}
				if (!isInShadow) {
					pixel.SetColor(object->Color() * max(0.0f, nhit.dot(shadowRay.Direction())) * light->brightness);
				}
				else
					pixel.SetColor(Vec3(0.0));
			}
			else
				pixel.SetColor(Vec3(0.3f));

			/*localP*/pixels[((image->Height()-1-j)*image->Width() + i)*4+0] = pixel.r;
			/*localP*/pixels[((image->Height()-1-j)*image->Width() + i)*4+1] = pixel.g;
			/*localP*/pixels[((image->Height()-1-j)*image->Width() + i)*4+2] = pixel.b;
			/*localP*/pixels[((image->Height()-1-j)*image->Width() + i)*4+3] = pixel.a;
		}
		
		void Render(int numThreads)
		{
			vector<thread> threads;
			for(int i = 0; i < numThreads; i++)
			{
				unsigned start = i * image->Width() / numThreads;
				unsigned end;
				if(i == numThreads-1)
					end = image->Width();
				else
					end = start + image->Width() / numThreads;
				threads.push_back(thread(&Raytracer::Thrender, this, start, end));
			}
			for(auto &thread : threads)
				thread.join();

			//for(int i = 0; i < image->Width()*image->Height()*4; i++)
			//	pixels[i] = localPixels[i];
		}
};