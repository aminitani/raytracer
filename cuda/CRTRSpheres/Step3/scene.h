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
		numSpheres = 5;
		spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		
		//spheres[0] = Sphere( Vec3(3, 1, -2), 3.0, Material(0, 0.5, 1.5, Vec3(1, 0, 1)) );
		//spheres[1] = Sphere( Vec3(-2, -1, 2), 2.0, Material(0.0, 0.1, 1.5, Vec3(1, 1, 1)) );
		//spheres[2] = Sphere( Vec3(0, -10005, 0), 10000, Material(0, 0.1, 1.5, Vec3(0, 1, 1)) );


		spheres[0] = Sphere(Vec3(0, -10004, 0), 10000, Material(0, .2, 1.5, Vec3(.6)));
		spheres[1] = Sphere(Vec3(0, 0, 0), 4, Material(.5, 1, 1.2, Vec3(1.00, 0.32, 0.36)));
		spheres[2] = Sphere(Vec3(5, -1, 5), 2, Material(0, 1, 1.5, Vec3(0.90, 0.76, 0.46)));
		spheres[3] = Sphere(Vec3(5, 0, -5), 3, Material(0, 1, 1.5, Vec3(0.65, 0.77, 0.97)));
		spheres[4] = Sphere(Vec3(-5.5, 0, 5), 3, Material(0, 1, 1.5, Vec3(0.90, 0.90, 0.90)));


		light = new Light(Vec3(0, 20, -30), 0.8);
		//light = new Light(Vec3(-3.0,10.0,3.0), .8);
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
};