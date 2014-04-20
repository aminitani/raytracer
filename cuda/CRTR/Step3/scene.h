#pragma once
#include "assets\object.h"
#include "assets\sphere.h"
#include "assets\triangle.h"
#include "assets\light.h"

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Scene
{
	Sphere *spheres;
	unsigned numSpheres;

	Light *light;
	//unsigned numLights;

	CUDA_CALLABLE_MEMBER Scene()
	{
		//casting to type * is bad in c, but c++ compiler needs it, sadface
		numSpheres = 1;
		spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		spheres[0] = Sphere(Vec3(1.0,1.0,-1.0), 3.0, Vec3(0.5,0.0,0.0));
		//spheres[0] = Sphere(Vec3(2, 1, -1), 3.0, Vec3(1, 0, 1));
		//spheres[1] = Sphere(Vec3(-1, -1, 1), 2.0, Vec3(1, 1, 0));

		light = new Light(Vec3(-3.0,3.0,3.0), 0.8);
	}

	CUDA_CALLABLE_MEMBER Scene(const Scene &scene)
	{
		this->light = new Light(*scene.light);

		this->numSpheres = scene.numSpheres;
		this->spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		for(int i = 0; i < numSpheres; i++)
			this->spheres[i] = Sphere(scene.spheres[i]);
	}
	
#ifdef __CUDACC__
	__device__~Scene()
	{
		cudaFree((void**)&spheres);
		cudaFree((void**)&light);
		light = NULL;
		spheres = NULL;
		numSpheres = 0;
	}
#else
	~Scene()
	{
		delete [] spheres;
		delete light;
		light = NULL;
		spheres = NULL;
		numSpheres = 0;
	}
#endif

//	CUDA_CALLABLE_MEMBER Scene Devicify()
//	{
//		//unnecessary work; mallocing, then cudamallocing :/
//		Scene ret(*this);
//		
//		//TODO: right here, replace the pointer to spheres and the pointer to light
//		//in the scene variable with newly allocated device pointers before passing
//		//the scene to the device in the trace call
//		Sphere *dSpheres;
//		CudaMalloc((void**) &(dSpheres), sizeof(Sphere)*scene.numSpheres);
//		CudaMemcpy(dSpheres, scene.spheres, sizeof(Sphere)*scene.numSpheres, cudaMemcpyHostToDevice);
//		scene.spheres = dSpheres;
//		
//		Light *dLight;
//		CudaMalloc((void**) &(dLight), sizeof(Light));
//		CudaMemcpy(dLight, scene.light, sizeof(Light), cudaMemcpyHostToDevice);
//		scene.light = dLight;
//	}
};