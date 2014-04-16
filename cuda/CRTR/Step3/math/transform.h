#ifndef TRANSFORM_H
#define TRANSFORM_H

#pragma once

#include "vec3.h"

using std::ostream;
using std::endl;

class Transform
{
	private:
		float contents[4][4];
		int rows;
		int cols;
	
	public:
		//static const Transform Identity
		Transform()
		{
			rows = 4;
			cols = 4;
			Identify();
		}
		
		//I demand an array of 16 floats organized row-major!
		Transform(float input[])
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
		Transform(const Transform &other)
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
		int GetCols() const { return cols; }
		
		//access height
		int GetRows() const { return rows; }
		
		//access index in matrix w/ const
		float GetIndex(int row, int column) const { return contents[row][column]; }
		
		//access index in matrix by ref
		float& operator()(int row, int column) { return contents[row][column]; }
		
		void Identify()
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
		
		const Vec3 Forward()
		{
			return Vec3((*this).GetIndex(0, 2), (*this).GetIndex(1, 2), (*this).GetIndex(2, 2));
		}
		
		const Vec3 Left()
		{
			return Vec3((*this).GetIndex(0, 0), (*this).GetIndex(1, 0), (*this).GetIndex(2, 0));
		}
		
		const Vec3 Up()
		{
			return Vec3((*this).GetIndex(0, 1), (*this).GetIndex(1, 1), (*this).GetIndex(2, 1));
		}
		
		const Vec3 Pos()
		{
			return Vec3((*this).GetIndex(3, 0), (*this).GetIndex(3, 1), (*this).GetIndex(3, 2));
		}

		void Translate(Vec3 vec)
		{
			contents[3][0] += vec.x;
			contents[3][1] += vec.y;
			contents[3][2] += vec.z;
		}
		
		Transform& operator=(const Transform& other)
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