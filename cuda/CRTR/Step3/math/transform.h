#ifndef TRANSFORM_H
#define TRANSFORM_H

#pragma once

#include "vec3.h"

using std::ostream;
using std::endl;
using std::cosf;
using std::sinf;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

struct Transform
{
	private:
		float contents[4][4];
		int rows;
		int cols;
	
	public:
		//static const Transform Identity
		CUDA_CALLABLE_MEMBER Transform()
		{
			rows = 4;
			cols = 4;
			Identify();
		}
		
		//I demand an array of 16 floats organized row-major!
		CUDA_CALLABLE_MEMBER Transform(float input[])
		{
			rows = 4;
			cols = 4;
			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < cols; j++)
				{
					contents[i][j] = (float)input[4 * i + j];
				}
			}
		}
		
		CUDA_CALLABLE_MEMBER static Transform TransformFromPos(Vec3 inPos)
		{
			Transform trans = Transform();
			trans(3, 0) = inPos.x;
			trans(3, 1) = inPos.y;
			trans(3, 2) = inPos.z;

			return trans;
		}
		
		//not tested
		CUDA_CALLABLE_MEMBER Transform(const Transform &other)
		{
			rows = 4;
			cols = 4;
			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < cols; j++)
				{
					contents[i][j] = other.GetIndex(i,j);
				}
			}
		}
		
		//access width
		CUDA_CALLABLE_MEMBER int GetCols() const { return cols; }
		
		//access height
		CUDA_CALLABLE_MEMBER int GetRows() const { return rows; }
		
		//access index in matrix w/ const
		CUDA_CALLABLE_MEMBER float GetIndex(int row, int column) const { return contents[row][column]; }
		
		//access index in matrix by ref
		CUDA_CALLABLE_MEMBER float& operator()(int row, int column) { return contents[row][column]; }
		
		CUDA_CALLABLE_MEMBER void Identify()
		{
			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < cols; j++)
				{
					if(i == j)
						contents[i][j] = 1;
					else//if(i != j)
						contents[i][j] = 0;
				}
			}
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Forward()
		{
			return Vec3((*this).GetIndex(2, 0), (*this).GetIndex(2, 1), (*this).GetIndex(2, 2));
		}
		
		CUDA_CALLABLE_MEMBER void SetForward(Vec3 in)
		{
			contents[2][0] = in.x;
			contents[2][1] = in.y;
			contents[2][2] = in.z;
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Left()
		{
			return Vec3((*this).GetIndex(0, 0), (*this).GetIndex(0, 1), (*this).GetIndex(0, 2));
		}
		
		CUDA_CALLABLE_MEMBER void SetLeft(Vec3 in)
		{
			contents[0][0] = in.x;
			contents[0][1] = in.y;
			contents[0][2] = in.z;
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Up()
		{
			return Vec3((*this).GetIndex(1, 0), (*this).GetIndex(1, 1), (*this).GetIndex(1, 2));
		}
		
		CUDA_CALLABLE_MEMBER void SetUp(Vec3 in)
		{
			contents[1][0] = in.x;
			contents[1][1] = in.y;
			contents[1][2] = in.z;
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Pos()
		{
			return Vec3((*this).GetIndex(3, 0), (*this).GetIndex(3, 1), (*this).GetIndex(3, 2));
		}
		
		CUDA_CALLABLE_MEMBER void SetPos(Vec3 in)
		{
			contents[3][0] = in.x;
			contents[3][1] = in.y;
			contents[3][2] = in.z;
		}

		CUDA_CALLABLE_MEMBER void Translate(Vec3 vec)
		{
			contents[3][0] += vec.x;
			contents[3][1] += vec.y;
			contents[3][2] += vec.z;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnXAroundSelf(float deg)
		{
			deg = deg * GR_PI / 180.0;

			Vec3 pos = Pos();
			Translate(pos * -1);

			Transform transformation = Transform();
			transformation(1, 1) = cos(deg);
			transformation(1, 2) = sin(deg);
			transformation(2, 1) = -sin(deg);
			transformation(2, 2) = cos(deg);

			(*this) *= transformation;

			Translate(pos);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnYAroundSelf(float deg)
		{
			deg = deg * GR_PI / 180.0;

			Vec3 pos = Pos();
			Translate(pos * -1);

			Transform transformation = Transform();
			transformation(0, 0) = cos(deg);
			transformation(0, 2) = -sin(deg);
			transformation(2, 0) = sin(deg);
			transformation(2, 2) = cos(deg);

			(*this) *= transformation;

			Translate(pos);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnZAroundSelf(float deg)
		{
			deg = deg * GR_PI / 180.0;

			Vec3 pos = Pos();
			Translate(pos * -1);

			Transform transformation = Transform();
			transformation(0, 0) = cos(deg);
			transformation(0, 1) = sin(deg);
			transformation(1, 0) = -sin(deg);
			transformation(1, 1) = cos(deg);

			(*this) *= transformation;

			Translate(pos);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnXAroundPoint(float deg, Vec3 point = Vec3())
		{
			deg = deg * GR_PI / 180.0;

			Translate(point * -1);

			Transform transformation = Transform();
			transformation(1, 1) = cos(deg);
			transformation(1, 2) = sin(deg);
			transformation(2, 1) = -sin(deg);
			transformation(2, 2) = cos(deg);

			(*this) *= transformation;

			Translate(point);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnYAroundPoint(float deg, Vec3 point = Vec3())
		{
			deg = deg * GR_PI / 180.0;

			Translate(point * -1);

			Transform transformation = Transform();
			transformation(0, 0) = cos(deg);
			transformation(0, 2) = -sin(deg);
			transformation(2, 0) = sin(deg);
			transformation(2, 2) = cos(deg);

			(*this) *= transformation;

			Translate(point);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnZAroundPoint(float deg, Vec3 point = Vec3())
		{
			deg = deg * GR_PI / 180.0;

			Translate(point * -1);

			Transform transformation = Transform();
			transformation(0, 0) = cos(deg);
			transformation(0, 1) = sin(deg);
			transformation(1, 0) = -sin(deg);
			transformation(1, 1) = cos(deg);

			(*this) *= transformation;

			Translate(point);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnAxisAroundSelf(float deg, Vec3 axis)
		{
			deg = deg * GR_PI / 180.0;

			Vec3 pos = Pos();
			Translate(pos * -1);

			float c = cos(deg);
			float s = sin(deg);
			float t = 1 - c;

			float x = axis.x;
			float y = axis.y;
			float z = axis.z;
			
			float trans[16] = {
				t*x*x+c,t*x*y+s*z,t*x*z-s*y,0.0,
				t*x*y-s*z,t*y*y+c,t*y*z+s*x,0.0,
				t*x*z+s*y,t*y*z-s*x,t*z*z+c,0.0,
				0.0,0.0,0.0,1.0};
			Transform transformation = Transform(trans);

			(*this) *= transformation;

			Translate(pos);

			return *this;
		}

		CUDA_CALLABLE_MEMBER Transform& RotateOnAxisAroundPoint(float deg, Vec3 axis, Vec3 point = Vec3())
		{
			deg = deg * GR_PI / 180.0;
			
			Translate(point * -1);

			float c = cos(deg);
			float s = sin(deg);
			float t = 1 - c;

			float x = axis.x;
			float y = axis.y;
			float z = axis.z;
			
			float trans[16] = {
				t*x*x+c,t*x*y-s*z,t*x*z+s*y,0.0,
				t*x*y+s*z,t*y*y+c,t*y*z-s*x,0.0,
				t*x*z-s*y,t*y*z+s*x,t*z*z+c,0.0,
				0.0,0.0,0.0,1.0};
			Transform transformation = Transform(trans);

			(*this) *= transformation;

			Translate(point);

			return *this;
		}
		
		CUDA_CALLABLE_MEMBER static Transform RotationOnYAroundOrigin(float deg)
		{
			deg = deg * GR_PI / 180.0;

			Transform transformation = Transform();
			transformation(0, 0) = cos(deg);
			transformation(0, 2) = -sin(deg);
			transformation(2, 0) = sin(deg);
			transformation(2, 2) = cos(deg);

			return transformation;
		}
		
		CUDA_CALLABLE_MEMBER static void TransformVec3(Vec3 &vec, Transform trans)
		{
			Vec3 original(vec);
			vec.x = original.x*trans.GetIndex(0,0) +
				original.y*trans.GetIndex(1,0) +
				original.z*trans.GetIndex(2,0) +
				trans.GetIndex(3,0);
			vec.y = original.x*trans.GetIndex(0,1) +
				original.y*trans.GetIndex(1,1) +
				original.z*trans.GetIndex(2,1) +
				trans.GetIndex(3,1);
			vec.z = original.x*trans.GetIndex(0,2) +
				original.y*trans.GetIndex(1,2) +
				original.z*trans.GetIndex(2,2) +
				trans.GetIndex(3,2);
		}

		CUDA_CALLABLE_MEMBER Transform& operator=(const Transform& other)
		{
			if(this == &other)
				return *this;
			for(int row = 0; row < 4; row++)
			{
				for(int col = 0; col < 4; col++)
				{
					contents[row][col] = other.GetIndex(row,col);
				}
			}
			
			return *this;
		}
		
		CUDA_CALLABLE_MEMBER Transform& operator*=(const Transform& other) // compound assignment
		{
			Transform output;
			for(int row = 0; row < 4; row++)
			{
				for(int col = 0; col < 4; col++)
				{
					float sum = 0;
					for(int k = 0; k < (*this).GetCols(); k++)
					{
						sum = sum + (*this).GetIndex(row, k) * other.GetIndex(k, col);
					}
					output(row,col) = sum;
				}
			}

			(*this) = output;
			return *this; // return the result by reference
		}
		
		CUDA_CALLABLE_MEMBER Transform operator*(const Transform& other) // compound assignment
		{
			Transform output;
			for(int row = 0; row < 4; row++)
			{
				for(int col = 0; col < 4; col++)
				{
					float sum = 0;
					for(int k = 0; k < (*this).GetCols(); k++)
					{
						sum = sum + (*this).GetIndex(row, k) * other.GetIndex(k, col);
					}
					output(row,col) = sum;
				}
			}

			return output; // return the result by reference
		}
};

//ostream& operator<<(ostream& Out, const Transform& Item);

// inline Transform operator*(Transform a, const Transform& b)
// {
	// Transform output;

	// for(int row = 0; row < 4; row++)
	// {
		// for(int col = 0; col < 4; col++)
		// {
			// float sum;
			// for(int k = 0; k < a.GetCols(); k++)
			// {
				// sum = sum + a.GetIndex(row, k) * b.GetIndex(k, col);
			// }
			// output(row,col) = sum;
		// }
	// }
	// return output;
// }

#endif