#include <stdio.h>
#include "testStruct.h"
#include "assets\camera.h"
#include "scene.h"
#include "math\ray.h"
#include "assets\image.h"

#define N 10
#define CollisionError 0.05

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

__global__ void Trace(float *pixels, float INFINITY, Camera camera, Scene *scene)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = camera.Height() - 1 - (blockIdx.y * blockDim.y + threadIdx.y);

	//this is the color to be contributed to the pixels
	Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);

	// compute primary ray direction
	Ray primaryRay;
	ComputePrimaryRay(i, j, camera.Width(), camera.Height(), &primaryRay, &camera);
	

	float minDist = INFINITY;
	Triangle *triangle = NULL;
	for (unsigned k = 0; k < scene->numTriangles; ++k) {
		float t0 = INFINITY;
		if ((scene->triangles[k]).Intersect(primaryRay, &t0)) {
			if (t0 < minDist) {
				triangle = &(scene->triangles[k]);
				minDist = t0; // update min distance
			}
		}
	}
	if (triangle != NULL) {
		// compute phit and nhit
		Vec3 pHit = primaryRay.Point() + primaryRay.Direction() * minDist; // point of intersection
		Vec3 nHit = triangle->Normal();

		// compute illumination
		Ray shadowRay;
		shadowRay.Point() = pHit + nHit * CollisionError;
		shadowRay.Direction() = (scene->light->position - pHit/*(pHit + CollisionError * nhit)*/).normalize();
		bool isInShadow = false;
		for (unsigned k = 0; k < scene->numTriangles; ++k) {
			float shadowMinDist = (scene->light->position - pHit).length();
			float t0 = INFINITY;
			if ((scene->triangles[k]).Intersect(shadowRay, &t0)) {
				if (t0 > 0 && t0 < shadowMinDist) {
					triangle = &(scene->triangles[k]);
					isInShadow = true;
					break;
				}
			}
		}
		if (!isInShadow) {
			pixel.SetColor(triangle->Color() * max(0.0f, nHit.dot(shadowRay.Direction())) * scene->light->brightness);
		}
		else {
			pixel.SetColor(Vec3(0));
		}
	}
	else
		pixel.SetColor(Vec3(0.3));
	


	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+0] = pixel.r;
	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+1] = pixel.g;
	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+2] = pixel.b;
	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+3] = pixel.a;
}



extern "C" {

	void CUDAThrender(float *pixels, float INFINITY, Camera camera, Scene scene)
	{
		//TODO: right here, replace the pointer to spheres and the pointer to light
		//in the scene variable with newly allocated device pointers before passing
		//the scene to the device in the trace call
		/*
		Sphere *dSpheres;
		cudaMalloc((void**) &(dSpheres), sizeof(Sphere)*scene.numSpheres);
		cudaMemcpy(dSpheres, scene.spheres, sizeof(Sphere)*scene.numSpheres, cudaMemcpyHostToDevice);
		scene.spheres = dSpheres;
		*/
		Triangle *dTriangles;
		cudaMalloc((void**) &(dTriangles), sizeof(Triangle)*scene.numTriangles);
		cudaMemcpy(dTriangles, scene.triangles, sizeof(Triangle)*scene.numTriangles, cudaMemcpyHostToDevice);
		scene.triangles = dTriangles;
		
		Light *dLight;
		cudaMalloc((void**) &(dLight), sizeof(Light));
		cudaMemcpy(dLight, scene.light, sizeof(Light), cudaMemcpyHostToDevice);
		scene.light = dLight;
		
		Scene *dScene;
		cudaMalloc((void**) &(dScene), sizeof(Scene));
		cudaMemcpy(dScene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

		dim3 block(8,8,1);
		dim3 grid(camera.Width()/block.x, camera.Height()/block.y, 1);
		Trace<<<grid, block>>>(pixels, INFINITY, camera, dScene);
		
		//red<<<grid, block>>>(pixels, ts, camera, scene);
	}


	void renderTest(float *out, int w, int h) {
		dim3 block(8,8,1);
		dim3 grid(w/block.x, h/block.y, 1);
		red<<<grid, block>>>(out,w,h);
	}


}