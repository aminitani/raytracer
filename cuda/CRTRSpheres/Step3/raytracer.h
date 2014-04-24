//TODO: multiple additive lights

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
//#include "./assets/triangle.h"
#include "./assets/light.h"
#include "./math/ray.h"
#include "scene.h"

using std::cout; using std::endl;
using std::vector;
using std::thread;

#define COLLISIONERROR .0001
#define MAXBOUNCES 5

class Raytracer
{
	private:
		float INFINITY;
		//Image *image;
		Camera *camera;//note that there are two cameras; childview's updates constantly, while this gets 'snapshots' every rendered frame
		Scene *scene;
		float *pixels;//the pixel buffer shared between the childview and this raytracer
		
		unsigned int m_width, m_height;

		Vec3 defaultColor;
		Vec3 BLACK;

		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point iterationDone;
		std::chrono::high_resolution_clock::time_point bufferCopied;
	
	public:
		float *buffer;//write to this, then push into pixels when the whole image is done

		Raytracer(float *inPixels, Camera inCamera, Scene inScene)
		{
			INFINITY = std::numeric_limits<float>::infinity();
			//image = new Image(width, height, "test.png");
			m_width = inCamera.Width(); m_height = inCamera.Height();
			camera = new Camera(Transform(), 0, 0, 0);
			*camera = inCamera;

			pixels = inPixels;
			buffer = new float[m_width*m_height*4];

			defaultColor = Vec3(0.3);
			BLACK = Vec3(0.0);
			scene = new Scene(inScene);
		}
		
		~Raytracer()
		{
			//delete image;
			//image = NULL;

			delete scene;
			scene = NULL;
			
			delete camera;
			camera = NULL;
			
			delete [] buffer;
			buffer = NULL;
			
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

		void ComputePrimaryRay(unsigned x, unsigned y, unsigned width, unsigned height, Ray *primaryRay, Camera *camera)
		{
			primaryRay->Point() = camera->orientation.Pos();
			Vec3 target =
			camera->GetViewPlane().ULCorner - camera->orientation.Left() * ( ((x+.5) / width) * 2*camera->GetViewPlane().hor )
			- camera->orientation.Up() * ( ((y+.5) / height) * 2*camera->GetViewPlane().vert );
			primaryRay->Direction() = (target - camera->orientation.Pos()).normalize();
		}

		void SaturateColor(Vec3 &color)
		{
			if(color.x > 1)
				color.x = 1;
			if(color.y > 1)
				color.y = 1;
			if(color.z > 1)
				color.z = 1;
		}

		float mix(const float &a, const float  &b, const float &mix)
		{
			return b * mix + a * (1.0 - mix);
		}

		Vec3 Trace(Ray &ray, int depth)
		{
			float minDist = INFINITY;
			Sphere *sphere = NULL;
			for (unsigned k = 0; k < scene->numSpheres; ++k) {
				float t0 = INFINITY;
				if ((scene->spheres[k]).Intersect(ray, &t0)) {
					if (t0 < minDist) {
						sphere = &(scene->spheres[k]);
						minDist = t0; // update min distance
					}
				}
			}
			if (sphere != NULL) {
				Vec3 surfaceColor = Vec3();

				// compute phit and nhit
				Vec3 pHit = ray.Point() + ray.Direction() * minDist; // point of intersection
				Vec3 nHit = sphere->Normal(pHit);

				// compute illumination
				bool inside = false;
				if(ray.Direction().dot(nHit) > 0)
				{
					nHit = nHit * -1;
					inside = true;
				}

				if ((sphere->GetMaterial().transparency > 0 || sphere->GetMaterial().reflection > 0) && depth < MAXBOUNCES)
				{
					float facingRatio = -ray.Direction().dot(nHit);
					float fresnelEffect = mix(pow(1 - facingRatio, 3), 1, 0.1);

					Vec3 reflectionDirection = ray.Direction() - nHit * 2 * ray.Direction().dot(nHit);
					reflectionDirection.normalize();

					Vec3 reflection = Trace( Ray(pHit + nHit * COLLISIONERROR, reflectionDirection), depth + 1 );
					Vec3 refraction = Vec3(0);

			
					if (sphere->GetMaterial().transparency > 0) {
						float eta = (inside) ? sphere->GetMaterial().IOR : 1 / sphere->GetMaterial().IOR; // are we inside or outside the surface?
						float cosi = -nHit.dot(ray.Direction());
						float k = 1 - eta * eta * (1 - cosi * cosi);
						Vec3 refractionDirection = ray.Direction() * eta + nHit * (eta *  cosi - sqrt(k));
						refractionDirection.normalize();
						refraction = Trace( Ray(pHit - nHit * COLLISIONERROR, refractionDirection), depth + 1 );
					}
					// the result is a mix of reflection and refraction (if the sphere is transparent)
					surfaceColor = (reflection * fresnelEffect + 
						refraction * (1 - fresnelEffect) * sphere->GetMaterial().transparency) * sphere->GetMaterial().color;
				}
				else//diffuse, just get the color now
				{
					Ray shadowRay;
					shadowRay.Point() = pHit;
					shadowRay.Direction() = (scene->light->position - pHit/*(pHit + COLLISIONERROR * nhit)*/).normalize();
					bool isInShadow = false;
					for (unsigned k = 0; k < scene->numSpheres; ++k) {
						float t0 = INFINITY;
						if ((scene->spheres[k]).Intersect(shadowRay, &t0)) {
							isInShadow = true;
							break;
						}
					}
					if (!isInShadow) {
						surfaceColor = sphere->GetMaterial().color * max(0.0f, nHit.dot(shadowRay.Direction())) * scene->light->Brightness( (scene->light->position - pHit).length() );
						SaturateColor(surfaceColor);
					}
					else {
						surfaceColor = Vec3(0);
					}
				}

				return surfaceColor;
			}
			else
				return Vec3(1.0);
		}

		void Thrender(unsigned int start, unsigned int end)
		{
			//unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
			//unsigned int j = camera.Height() - 1 - (blockIdx.y * blockDim.y + threadIdx.y);

			for(int i = start; i < end; i++) {
				for(int j = 0; j < m_height; j++) {
					Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);

					// compute primary ray direction
					Ray primaryRay;
					ComputePrimaryRay(i, j, camera->Width(), camera->Height(), &primaryRay, camera);
	
					pixel.SetColor(Trace(primaryRay, 0));

					buffer[((m_height-1-j)*m_width + i)*4+0] = pixel.r;
					buffer[((m_height-1-j)*m_width + i)*4+1] = pixel.g;
					buffer[((m_height-1-j)*m_width + i)*4+2] = pixel.b;
					buffer[((m_height-1-j)*m_width + i)*4+3] = pixel.a;
				}
			}
		}
		
		void Render(int numThreads, Scene scene, Camera newCam)
		{
			*camera = newCam;
			m_width = camera->Width();
			m_height = camera->Height();
			*this->scene = scene;
			start = std::chrono::high_resolution_clock::now();
			
			vector<thread> threads;
			for(int i = 0; i < numThreads; i++)
			{
				unsigned start = i * (m_width / numThreads);
				unsigned end;
				if(i == numThreads-1)
					end = m_width;
				else
					end = start + m_width / numThreads;
				threads.push_back(thread(&Raytracer::Thrender, this, start, end));
			}
			for(auto &thread : threads)
				thread.join();
			

			iterationDone = std::chrono::high_resolution_clock::now();
			
			//for(int i = 0; i < image->Width()*image->Height()*4; i++)
			//	pixels[i] = buffer[i];

			bufferCopied = std::chrono::high_resolution_clock::now();
			
			unsigned long long traceTime = (float)(iterationDone - start).count() / 1000;
			unsigned long long copyTime = (float)(bufferCopied - iterationDone).count() / 1000;
			int x = 0;
		}
};