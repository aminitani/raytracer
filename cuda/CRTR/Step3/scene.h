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
	Triangle *triangles;
	unsigned numTriangles;

	Light *light;
	//unsigned numLights;

	CUDA_CALLABLE_MEMBER Scene(unsigned numTris, Vec3 *triVerts, Vec3 *triNorms)
	{
		numTriangles = numTris;
		triangles = (Triangle *) malloc (numTriangles * sizeof(Triangle));
		
		for(unsigned i=0; i<numTriangles; i++)
			triangles[i] = Triangle(triVerts[i*3], triVerts[i*3+1], triVerts[i*3+2], triNorms[i], Material(0, 0, 1.5, Vec3(0.0,0.0,0.4)));
		
		light = new Light(Vec3(0, 20, -30), 0.8);
		//light = new Light(Vec3(-3.0,10.0,3.0), .8);
	}

	CUDA_CALLABLE_MEMBER Scene(const Scene &scene)
	{
		this->light = new Light(*scene.light);
		
		this->numTriangles = scene.numTriangles;
		this->triangles = (Triangle *) malloc (numTriangles * sizeof(Triangle));
		for(int i = 0; i < numTriangles; i++)
			this->triangles[i] = Triangle(scene.triangles[i]);
	}
	
#ifdef __CUDACC__
	__device__~Scene()
	{
		cudaFree((void**)&triangles);
		cudaFree((void**)&light);
		light = NULL;
		triangles = NULL;
		numTriangles = 0;
	}
#else
	~Scene()
	{
		delete [] triangles;
		delete light;
		light = NULL;
		triangles = NULL;
		numTriangles = 0;
	}
#endif
};