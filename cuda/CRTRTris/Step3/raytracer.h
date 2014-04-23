//TODO: multiple additive lights
//TODO: inverse square light falloff

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <thread>
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
using std::thread;

#define CollisionError 0.05

class Raytracer
{
	private:
		float INFINITY;
		Image *image;
		Camera *camera;//note that there are two cameras; childview's updates constantly, while this gets 'snapshots' every rendered frame
		vector<Object *> objects;
		Light *light;
		float *pixels;//the pixel buffer shared between the childview and this raytracer
		float *localPixels;//write to this, then push into pixels when the whole image is done

		Vec3 defaultColor;
		Vec3 BLACK;

		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point iterationDone;
		std::chrono::high_resolution_clock::time_point bufferCopied;

		void ConstructScene() {
			light = new Light(Vec3(-3.0,3.0,3.0), 0.8);
			Sphere *sphere = new Sphere(Vec3(1.0,1.0,-1.0), 3.0, Vec3(0.5,0.0,0.0));
			objects.push_back(sphere);
		}
	
	public:
		Raytracer(int width, int height, float *inPixels, Camera inCamera)
		{
			INFINITY = std::numeric_limits<float>::infinity();
			image = new Image(width, height, "test.png");
			camera = new Camera(Transform(), 0, 0, 0);
			*camera = inCamera;

			pixels = inPixels;
			localPixels = new float[width*height*4];
			ConstructScene();

			defaultColor = Vec3(0.3);
			BLACK = Vec3(0.0);
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

			delete [] localPixels;
			localPixels = NULL;
			
			//don't delete pixels; childview's job
			pixels = NULL;
		}

		Camera *GetCamera() {return camera;}

//		void Thrender(unsigned start, unsigned end, ViewPlane &vp)
//		{
//			Point pHit;
//			Normal nHit;
//			//this is the color to be contributed to the pixels
//			Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);
//			Ray primaryRay;
//			float minDist;
//
//			for (unsigned i = start; i < end; ++i)
//			{
//				for (unsigned j = 0; j < image->Height(); ++j)
//				{
//					Trace(i, j, pHit, nHit, pixel, primaryRay, minDist, vp);
//				}
//			}
//		}
//
//		void Trace(const int &i, const int &j, Point &pHit, Normal &nHit, Pixel &pixel, Ray &primaryRay, float &minDist, ViewPlane &vp)
//		{
//			// compute primary ray direction
//			ComputePrimaryRay(i, j, image->Width(), image->Height(), &primaryRay, vp);
//			minDist = INFINITY;
//			Object *object = NULL;
//			for (unsigned k = 0; k < objects.size(); ++k) {
////				if (Intersect(objects[k], primRay, &pHit, &nHit)) {
//				float t0 = INFINITY;
//				if ((*objects[k]).Intersect(primaryRay, &t0)) {
//					// float distance = Vec3::Distance(eyePosition, pHit);
//					if (t0 < minDist) {
//						object = objects[k];
//						minDist = t0; // update min distance
//					}
//				}
//			}
//			if (object != NULL) {
//				// compute phit and nhit
//				pHit = primaryRay.Point() + primaryRay.Direction() * minDist; // point of intersection
//				nHit = object->Normal(pHit);
//
//				// compute illumination
//				Ray shadowRay;
//				shadowRay.Point() = pHit;
//				shadowRay.Direction() = (light->position - pHit/*(pHit + CollisionError * nhit)*/).normalize();
//				bool isInShadow = false;
//				for (unsigned k = 0; k < objects.size(); ++k) {
////					if (Intersect(objects[k], shadowRay)) {
//					float t0 = INFINITY;
//					if ((*objects[k]).Intersect(shadowRay, &t0)) {
//						isInShadow = true;
//						break;
//					}
//				}
//				if (!isInShadow) {
//					pixel.SetColor(object->Color() * max(0.0f, nHit.dot(shadowRay.Direction())) * light->brightness);
//				}
//				else
//					pixel.SetColor(BLACK);
//			}
//			else
//				pixel.SetColor(defaultColor);
//
//			pixels[((image->Height()-1-j)*image->Width() + i)*4+0] = pixel.r;
//			pixels[((image->Height()-1-j)*image->Width() + i)*4+1] = pixel.g;
//			pixels[((image->Height()-1-j)*image->Width() + i)*4+2] = pixel.b;
//			pixels[((image->Height()-1-j)*image->Width() + i)*4+3] = pixel.a;
//		}
		
		void Render(int numThreads, Camera newCam)
		{
			*camera = newCam;

			start = std::chrono::high_resolution_clock::now();
			
			//vector<thread> threads;
			//for(int i = 0; i < numThreads; i++)
			//{
			//	unsigned start = i * image->Width() / numThreads;
			//	unsigned end;
			//	if(i == numThreads-1)
			//		end = image->Width();
			//	else
			//		end = start + image->Width() / numThreads;
			//	threads.push_back(thread(&Raytracer::Thrender, this, start, end, vp));
			//}
			//for(auto &thread : threads)
			//	thread.join();
			

			iterationDone = std::chrono::high_resolution_clock::now();
			
			//for(int i = 0; i < image->Width()*image->Height()*4; i++)
			//	pixels[i] = localPixels[i];

			bufferCopied = std::chrono::high_resolution_clock::now();
			
			unsigned long long traceTime = (float)(iterationDone - start).count() / 1000;
			unsigned long long copyTime = (float)(bufferCopied - iterationDone).count() / 1000;
			int x = 0;
		}
};