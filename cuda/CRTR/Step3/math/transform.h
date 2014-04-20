#ifndef TRANSFORM_H
#define TRANSFORM_H

#pragma once

#include "vec3.h"

using std::ostream;
using std::endl;

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
			return Vec3((*this).GetIndex(0, 2), (*this).GetIndex(1, 2), (*this).GetIndex(2, 2));
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Left()
		{
			return Vec3((*this).GetIndex(0, 0), (*this).GetIndex(1, 0), (*this).GetIndex(2, 0));
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Up()
		{
			return Vec3((*this).GetIndex(0, 1), (*this).GetIndex(1, 1), (*this).GetIndex(2, 1));
		}
		
		CUDA_CALLABLE_MEMBER const Vec3 Pos()
		{
			return Vec3((*this).GetIndex(3, 0), (*this).GetIndex(3, 1), (*this).GetIndex(3, 2));
		}

		CUDA_CALLABLE_MEMBER void Translate(Vec3 vec)
		{
			contents[3][0] += vec.x;
			contents[3][1] += vec.y;
			contents[3][2] += vec.z;
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
		
		// Transform& operator*=(const Transform& other) // compound assignment
		// {
			// Transform output;
			// for(int row = 0; row < 4; row++)
			// {
				// for(int col = 0; col < 4; col++)
				// {
					// float sum;
					// for(int k = 0; k < (*this).GetCols(); k++)
					// {
						// sum = sum + (*this).GetIndex(row, k) * other.GetIndex(k, col);
					// }
					// output(row,col) = sum;
				// }
			// }

			// (*this) = output;
			// return *this; // return the result by reference
		// }
};

ostream& operator<<(ostream& Out, const Transform& Item);

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