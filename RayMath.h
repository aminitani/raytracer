//By Siddhartha Chaudhuri
//http://fuzzyphoton.tripod.com/howtowrt.htm

namespace RayMath
{
	#define LARGE_VAL 1e10

	double fdiv(double a, double b);
	int sign(double x);
	bool calc_root_of_linear(double & ret, double c1, double c0);

	// returns a / b (0 / 0 = 0, overflow = LARGE_VAL with correct sign)
	double fdiv(double a, double b)
	{
		if (b == 0)
		{
			if (a == 0) return 0;
			else return LARGE_VAL * sign(a);
		}
		else
		{
			if (a == 0) return 0;
			else
			{
				if ((a + b) == a) return LARGE_VAL * sign(a) * sign(b);
				else return a / b;
			}
		}
	}

	// sign function
	int sign(double x)
	{
		return (x == 0 ? 0 : (x < 0 ? -1 : 1));
	}

	// root of linear equation c1 * x + c0 = 0
	// if successful, returns true and places value in ret = x
	bool calc_root_of_linear(double & ret, double c1, double c0)
	{
		if (c1 == 0)
			return false;
		else
		{
			ret = fdiv(-c0, c1);
			return true;
		}
	}
}
