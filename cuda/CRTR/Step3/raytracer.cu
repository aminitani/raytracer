#include <stdio.h>
#include "testStruct.h"
#include "assets\camera.h"
#include "scene.h"
#include "math\ray.h"
#include "assets\image.h"
#include "math_constants.h"

#define N 10
#define MAXBOUNCES 5
#define COLLISIONERROR .0001

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

__device__ float mix(const float &a, const float  &b, const float &mix)
{
	return b * mix + a * (1.0 - mix);
}

__device__ void SaturateColor(Vec3 &color)
{
	if(color.x > 1)
		color.x = 1;
	if(color.y > 1)
		color.y = 1;
	if(color.z > 1)
		color.z = 1;
}

__device__ void ComputePrimaryRay(unsigned x, unsigned y, unsigned width, unsigned height, Ray *primaryRay, Camera *camera)
{
	primaryRay->Point() = camera->orientation.Pos();
	Vec3 target =
	camera->GetViewPlane().ULCorner - camera->orientation.Left() * ( ((x+.5) / width) * 2*camera->GetViewPlane().hor )
	- camera->orientation.Up() * ( ((y+.5) / height) * 2*camera->GetViewPlane().vert );
	primaryRay->Direction() = (target - camera->orientation.Pos()).normalize();
}

template <int depth>
__device__ Vec3 Trace(Ray &ray, Scene *scene)
{
	float minDist = CUDART_INF_F;
	Sphere *sphere = NULL;
	for (unsigned k = 0; k < scene->numSpheres; ++k) {
		float t0 = CUDART_INF_F;
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

			Vec3 reflection = Trace<depth+1>( Ray(pHit + nHit * COLLISIONERROR, reflectionDirection), scene );
			Vec3 refraction = Vec3(0);

			
			if (sphere->GetMaterial().transparency > 0) {
				float eta = (inside) ? sphere->GetMaterial().IOR : 1 / sphere->GetMaterial().IOR; // are we inside or outside the surface?
				float cosi = -nHit.dot(ray.Direction());
				float k = 1 - eta * eta * (1 - cosi * cosi);
				Vec3 refractionDirection = ray.Direction() * eta + nHit * (eta *  cosi - sqrt(k));
				refractionDirection.normalize();
				refraction = Trace<depth + 1>( Ray(pHit - nHit * COLLISIONERROR, refractionDirection), scene );
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
				float t0 = CUDART_INF_F;
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

template<>
__device__ Vec3 Trace<MAXBOUNCES>(Ray &ray, Scene *scene)
{
	float minDist = CUDART_INF_F;
	Sphere *sphere = NULL;
	for (unsigned k = 0; k < scene->numSpheres; ++k) {
		float t0 = CUDART_INF_F;
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

		Ray shadowRay;
		shadowRay.Point() = pHit;
		shadowRay.Direction() = (scene->light->position - pHit/*(pHit + COLLISIONERROR * nhit)*/).normalize();
		bool isInShadow = false;
		for (unsigned k = 0; k < scene->numSpheres; ++k) {
			float t0 = CUDART_INF_F;
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

		return surfaceColor;
	}
	else
		return Vec3(0.3);
}

__global__ void PrepareTrace(float *pixels, Camera camera, Scene *scene)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = camera.Height() - 1 - (blockIdx.y * blockDim.y + threadIdx.y);

	//this is the color to be contributed to the pixels
	Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);

	// compute primary ray direction
	Ray primaryRay;
	ComputePrimaryRay(i, j, camera.Width(), camera.Height(), &primaryRay, &camera);
	
	pixel.SetColor(Trace<0>(primaryRay, scene));

	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+0] = pixel.r;
	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+1] = pixel.g;
	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+2] = pixel.b;
	pixels[((camera.Height()-1-j)*camera.Width() + i)*4+3] = pixel.a;
}



extern "C" {

	void CUDAThrender(float *pixels, Camera camera, Scene scene)
	{
		//TODO: right here, replace the pointer to spheres and the pointer to light
		//in the scene variable with newly allocated device pointers before passing
		//the scene to the device in the trace call
		Sphere *dSpheres;
		cudaMalloc((void**) &(dSpheres), sizeof(Sphere)*scene.numSpheres);
		cudaMemcpy(dSpheres, scene.spheres, sizeof(Sphere)*scene.numSpheres, cudaMemcpyHostToDevice);
		scene.spheres = dSpheres;
		
		Light *dLight;
		cudaMalloc((void**) &(dLight), sizeof(Light));
		cudaMemcpy(dLight, scene.light, sizeof(Light), cudaMemcpyHostToDevice);
		scene.light = dLight;
		
		Scene *dScene;
		cudaMalloc((void**) &(dScene), sizeof(Scene));
		cudaMemcpy(dScene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);

		dim3 block(8,8,1);
		dim3 grid(camera.Width()/block.x, camera.Height()/block.y, 1);
		PrepareTrace<<<grid, block>>>(pixels, camera, dScene);
	}


	void renderTest(float *out, int w, int h) {
		dim3 block(8,8,1);
		dim3 grid(w/block.x, h/block.y, 1);
		red<<<grid, block>>>(out,w,h);
	}


}