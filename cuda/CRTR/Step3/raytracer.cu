#include <stdio.h>
#define N 10

class MyClass {
public:
	float x;
	__device__ MyClass() {
		x = .5;
	}
};

//struct ViewPlane
//{
//	ViewPlane(Camera *camera)
//	{
//		//these are half the viewplane dimensions
//		//TODO: tan is a bad function. works for most reasonable values though.
//		vert = (float) tan(camera->Fovy() / 2.0) * camera->VPD();
//		hor = vert * camera->ARatio();
//		VPCenter = camera->orientation.Pos() + ( camera->orientation.Forward() * camera->VPD() );
//		ULCorner = VPCenter + camera->orientation.Left() * hor + camera->orientation.Up() * vert;
//	}
//	float vert, hor;
//	Point VPCenter, ULCorner;
//};

__global__ void add( int *a, int *b, int *c ) {
	 int tid = blockIdx.x; // handle the data at this index
	 if (tid < N)
	 c[tid] = a[tid] + b[tid];
}

__global__ void red( float *out, int w, int h) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	MyClass mc;

	unsigned int i = (y * w + x) * 4;
	out[i] = mc.x;
	out[i + 1] = (float)y / h;
	out[i + 2] = (float)x / w;
	out[i + 3] = 1.0;
}

//__device__ void ComputePrimaryRay(unsigned x, unsigned y, unsigned width, unsigned height, Ray *primaryRay, ViewPlane &vp)
//{
//	primaryRay->Point() = camera->orientation.Pos();
//	Point target =
//	vp.ULCorner - camera->orientation.Left() * ( ((x+.5) / width) * 2*vp.hor )
//	- camera->orientation.Up() * ( ((y+.5) / height) * 2*vp.vert );
//	primaryRay->Direction() = (target - camera->orientation.Pos()).normalize();
//}
//
//__global__ void Trace(float *pixels, int width, int height, ViewPlane &vp)
//{
//	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
//
//	//this is the color to be contributed to the pixels
//	Pixel pixel(0.8f, 0.8f, 0.8f, 0.5f);
//
//	// compute primary ray direction
//	Ray primaryRay;
//	ComputePrimaryRay(i, j, width, height, &primaryRay, vp);
//	float minDist = INFINITY;
//	Object *object = NULL;
//	for (unsigned k = 0; k < objects.size(); ++k) {
//		float t0 = INFINITY;
//		if ((*objects[k]).Intersect(primaryRay, &t0)) {
//			if (t0 < minDist) {
//				object = objects[k];
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
//	pixels[((height-1-j)*width + i)*4+0] = pixel.r;
//	pixels[((height-1-j)*width + i)*4+1] = pixel.g;
//	pixels[((height-1-j)*width + i)*4+2] = pixel.b;
//	pixels[((height-1-j)*width + i)*4+3] = pixel.a;
//}



extern "C" {

	void CUDAThrender(float *pixels, int w, int h, float4 camInfo)
	{
		//ViewPlane vp(camera);

		//dim3 block(8,8,1);
		//dim3 grid(w/block.x, h/block.y, 1);
		//trace<<<grid, block>>>(pixels, w, h, vp);
	}


	void renderTest(float *out, int w, int h) {
		dim3 block(8,8,1);
		dim3 grid(w/block.x, h/block.y, 1);
		red<<<grid, block>>>(out,w,h);
	}


}