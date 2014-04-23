#include "StdAfx.h"

#include "math\transform.h"
#include <iostream>

//override output operator
ostream& operator<<(ostream& Out, const Transform& Item)
{
	for(int row = 0; row < Item.GetRows(); row++)
	{
		Out << "[";
		for(int column = 0; column < Item.GetCols() - 1; column++)
		{
			Out << Item.GetIndex(row, column) << " ";
		}
		Out << Item.GetIndex(row, Item.GetCols() - 1);
		Out << "]" << endl;
	}
	return Out;
}