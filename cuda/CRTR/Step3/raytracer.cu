#include <stdio.h>
#include "testStruct.h"
#include "assets\camera.h"
#include "scene.h"
#include "math\ray.h"
#include "assets\image.h"

#define N 10

__global__ void add( int *a, int *b, int *c ) {
	 int tid = blockIdx.x; // handle the data at this index
	 if (tid < N)
	 c[tid] = a[tid] + b[tid];
}

__global__ void red( float *out, int w, int h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int i = (y * w + x) * 4;
	out[i] = 1.0;
	out[i + 1] = (float)y / h;
	out[i + 2] = (float)x / w;
	out[i + 3] = 1.0;
}

__global__ void red( float *out, TestStruct ts, Camera camera, Scene scene) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned int i = (y * camera.Width() + x) * 4;
	out[i] = ts.r;
	out[i + 1] = ts.g;
	out[i + 2] = ts.b;
	out[i + 3] = 1.0;
}

__device__ void ComputePrimaryRay(unsigned x, unsigned y, unsigned width, unsigned height, Ray *primaryRay, Camera *camera)
{
	primaryRay->Point() = camera->orientation.Pos();
	Vec3 target =
	camera->GetViewPlane().ULCorner - camera->orientation.Left() * ( ((x+.5) / width) * 2*camera->GetViewPlane().hor )
	- camera->orientation.Up() * ( ((y+.5) / height) * 2*camera->GetViewPlane().vert );
	primaryRay->Direction() = (target - camera->orientation.Pos()).normalize();
}

__global__ void Trace(float *pixels, float INFINITY, Camera camera, Scene scene)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	//this is the color to be contributed to the pixels
	Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);

	// compute primary ray direction
	Ray primaryRay;
	ComputePrimaryRay(i, j, camera.Width(), camera.Height(), &primaryRay, &camera);
	
//	float minDist = INFINITY;
//	Sphere *sphere = NULL;
//	for (unsigned k = 0; k < scene.numSpheres; ++k) {
//		float t0 = INFINITY;
//		if ((scene.spheres[k]).Intersect(primaryRay, &t0)) {
//			if (t0 < minDist) {
//				sphere = &(scene.spheres[k]);
//				minDist = t0; // update min distance
//			}
//		}
//	}
//	if (object != NULL) {
//		// compute phit and nhit
//		Vec3 pHit = primaryRay.Point() + primaryRay.Direction() * minDist; // point of intersection
//		Vec3 nHit = object->Normal(pHit);
//
//		// compute illumination
//		Ray shadowRay;
//		shadowRay.Point() = pHit;
//		shadowRay.Direction() = (light->position - pHit/*(pHit + CollisionError * nhit)*/).normalize();
//		bool isInShadow = false;
//		for (unsigned k = 0; k < objects.size(); ++k) {
//			float t0 = INFINITY;
//			if ((*objects[k]).Intersect(shadowRay, &t0)) {
//				isInShadow = true;
//				break;
//			}
//		}
//		if (!isInShadow) {
//			pixel.SetColor(object->Color() * max(0.0f, nHit.dot(shadowRay.Direction())) * light->brightness);
//		}
//		else
//			pixel.SetColor(BLACK);
//	}
//	else
//		pixel.SetColor(defaultColor);
//
//	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+0] = pixel.r;
//	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+1] = pixel.g;
//	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+2] = pixel.b;
//	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+3] = pixel.a;

	//test crap
	unsigned int k = (j * camera.Width() + i) * 4;
	pixels[k] = 1.0;
	pixels[k + 1] = (float)i / camera.Width();
	pixels[k + 2] = (float)j / camera.Height();
	pixels[k + 3] = 1.0;
}



extern "C" {

	void CUDAThrender(float *pixels, float INFINITY, Camera camera, Scene scene)
	{
		dim3 block(8,8,1);
		dim3 grid(camera.Width()/block.x, camera.Height()/block.y, 1);
		Trace<<<grid, block>>>(pixels, INFINITY, camera, scene);
		
		//red<<<grid, block>>>(pixels, ts, camera, scene);
	}


	void renderTest(float *out, int w, int h) {
		dim3 block(8,8,1);
		dim3 grid(w/block.x, h/block.y, 1);
		red<<<grid, block>>>(out,w,h);
	}


}