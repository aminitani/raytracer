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
	//Sphere *spheres;
	//unsigned numSpheres;

	Triangle *triangles;
	unsigned numTriangles;

	Light *light;
	//unsigned numLights;

	CUDA_CALLABLE_MEMBER Scene(unsigned numTris, Vec3 *triVerts, Vec3 *triNorms)
	{
		/*
		numTris = 1; //you can easily limit how many tris are drawn in this way
		triVerts[0] = Vec3(-1.0,1.0,1.0);
		triVerts[1] = Vec3(1.0,1.0,1.0);
		triVerts[2] = Vec3(-1.0,1.0,-1.0);
		*/

		/*
		//casting to type * is bad in c, but c++ compiler needs it, sadface
		numSpheres = 3;
		spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		//spheres[0] = Sphere(Vec3(1.0,1.0,-1.0), 3.0, Vec3(0.5,0.0,0.0));
		spheres[0] = Sphere(Vec3(3, 1, -2), 3.0, Vec3(1, 0, 1));
		spheres[1] = Sphere(Vec3(-2, -1, 2), 2.0, Vec3(1, 1, 0));
		spheres[2] = Sphere(Vec3(0, -10005, 0), 10000, Vec3(0, 1, 1));
		*/

		//numTriangles = 12;
		numTriangles = numTris;
		triangles = (Triangle *) malloc (numTriangles * sizeof(Triangle));
		
		for(unsigned i=0; i<numTriangles; i++) {
			triangles[i] = Triangle(triVerts[i*3], triVerts[i*3+1], triVerts[i*3+2], triNorms[i], Vec3(0.0,0.0,0.4));
		}
		//Cube----------------------------------------------------------------------------------------------------------
		/*			  8------7    
					 /|     /|    
					4-+----3 |    
					| |    | |    
					| 5----+-6    
					|/     |/     
					1------2          *//*
		Vec3 trans = Vec3(-1.0,-1.0,1.0);
		Vec3 v1 = Vec3(0.0,0.0,0.0) + trans;
		Vec3 v2 = Vec3(2.0,0.0,0.0) + trans;
		Vec3 v3 = Vec3(2.0,2.0,0.0) + trans;
		Vec3 v4 = Vec3(0.0,2.0,0.0) + trans;
		Vec3 v5 = Vec3(0.0,0.0,-2.0) + trans;
		Vec3 v6 = Vec3(2.0,0.0,-2.0) + trans;
		Vec3 v7 = Vec3(2.0,2.0,-2.0) + trans;
		Vec3 v8 = Vec3(0.0,2.0,-2.0) + trans;

		//Front
		triangles[0] = Triangle(v1, v2, v3, Vec3(0.0,0.0,0.4));
		triangles[1] = Triangle(v1, v3, v4, Vec3(0.0,0.0,0.4));
		//Right
		triangles[2] = Triangle(v2, v6, v7, Vec3(0.0,0.0,0.4));
		triangles[3] = Triangle(v2, v7, v3, Vec3(0.0,0.0,0.4));
		//Left
		triangles[4] = Triangle(v5, v1, v4, Vec3(0.0,0.0,0.4));
		triangles[5] = Triangle(v5, v4, v8, Vec3(0.0,0.0,0.4));
		//Bottom
		triangles[6] = Triangle(v5, v6, v2, Vec3(0.0,0.0,0.4));
		triangles[7] = Triangle(v5, v2, v1, Vec3(0.0,0.0,0.4));
		//Top
		triangles[8] = Triangle(v4, v3, v7, Vec3(0.0,0.0,0.4));
		triangles[9] = Triangle(v4, v7, v8, Vec3(0.0,0.0,0.4));
		//Back
		triangles[10] = Triangle(v6, v5, v8, Vec3(0.0,0.0,0.4));
		triangles[11] = Triangle(v6, v8, v7, Vec3(0.0,0.0,0.4));
		//--------------------------------------------------------------------------------------------------------------
		*/
		light = new Light(Vec3(-30.0,30.0,30.0), 0.8);
	}

	CUDA_CALLABLE_MEMBER Scene(const Scene &scene)
	{
		this->light = new Light(*scene.light);
		/*
		this->numSpheres = scene.numSpheres;
		this->spheres = (Sphere *) malloc (numSpheres * sizeof(Sphere));
		for(int i = 0; i < numSpheres; i++)
			this->spheres[i] = Sphere(scene.spheres[i]);
		*/
		this->numTriangles = scene.numTriangles;
		this->triangles = (Triangle *) malloc (numTriangles * sizeof(Triangle));
		for(int i = 0; i < numTriangles; i++)
			this->triangles[i] = Triangle(scene.triangles[i]);
	}

	
#ifdef __CUDACC__
	__device__~Scene()
	{
		//cudaFree((void**)&spheres);
		cudaFree((void**)&triangles);
		cudaFree((void**)&light);
		light = NULL;
		//spheres = NULL;
		triangles = NULL;
		//numSpheres = 0;
		numTriangles = 0;
	}
#else
	~Scene()
	{
		//delete [] spheres;
		delete [] triangles;
		delete light;
		light = NULL;
		//spheres = NULL;
		triangles = NULL;
		//numSpheres = 0;
		numTriangles = 0;
	}
#endif
};