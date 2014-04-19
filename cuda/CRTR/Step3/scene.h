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

	Scene()
	{
		//casting to type * is bad in c, but c++ compiler needs it, sadface
		numSpheres = 2;
		spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		spheres[0] = Sphere(Vec3(2, 1, -1), 3.0, Vec3(1, 0, 1));
		spheres[1] = Sphere(Vec3(-1, -1, 1), 2.0, Vec3(1, 1, 0));

		light = new Light(Vec3(-3.0,3.0,3.0), 0.8);
	}

	Scene(const Scene &scene)
	{
		this->light = new Light(*scene.light);

		this->numSpheres = scene.numSpheres;
		spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		for(int i = 0; i < numSpheres; i++)
			this->spheres[i] = Sphere(scene.spheres[i]);
	}

	~Scene()
	{
		delete [] spheres;
		delete light;
		light = NULL;
		spheres = NULL;
		numSpheres = 0;
	}
};