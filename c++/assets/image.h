#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include "../png/lodepng.h"
#include "../math/vec3.h"

using std::string;
using std::vector;
using std::min;

class Pixel
{
public:
	float r, g, b, a;
	
	Pixel(float rr, float gg, float bb, float aa) : r(rr), g(gg), b(bb), a(aa) {}
	
	void SetColor(Vec3 color)
	{
		r = color.x;
		g = color.y;
		b = color.z;
		a = 1.0f;
	}
	
	void SetColor(float rr, float gg, float bb, float aa)
	{
		r = rr;
		g = gg;
		b = bb;
		a = aa;
	}
	
	Pixel operator + (const Pixel &p) const { return Pixel(r + p.r, g + p.g, b + p.b, a + p.a); }
	Pixel operator * (const unsigned int &i) const { return Pixel(r * i, g * i, b * i, a * i); }
	Pixel operator / (const unsigned int &i) const { return Pixel(r / i, g / i, b / i, a / i); }
};

class ImagePixel
{
private:
	Pixel conglomeratePixel = Pixel((float)0,(float)0,(float)0,(float)0);
	unsigned int contributions = 0;
	
public:
	ImagePixel() {}
	
	Pixel GetPixel() {return conglomeratePixel;}
	void Contribute(Pixel pixel)
	{
		if(contributions == 0)
			conglomeratePixel = pixel;
		else
			conglomeratePixel = (conglomeratePixel * contributions + pixel) / (contributions+1);
		contributions++;
	}
};

class Image
{
private:
	unsigned int width;
	unsigned int height;
	string name;
	vector<ImagePixel> imagePixels;
	
public:
	Image(unsigned int inWidth, unsigned int inHeight)
	{
		name = "rayOutput.png";
		width = inWidth; height = inHeight;
		imagePixels.resize(width * height);
	}
	
	Image(unsigned int inWidth, unsigned int inHeight, string inName)
	{
		if(inName.substr(inName.size()-4, 4)==".png")
			name = inName;
		else
			name = "rayOutput.png";
		
		width = inWidth; height = inHeight;
		imagePixels.resize(width * height);
	}
	
	unsigned int Width() {return width;}
	unsigned int Height() {return height;}
	ImagePixel& operator()(unsigned int x, unsigned int y) { return imagePixels[width * y + x]; }
	
	void Contribute(unsigned int x, unsigned int y, Pixel contribution)
	{
		(*this)(x,y).Contribute(contribution);
	}
	
	void Save()
	{
		//generate some image
		//The image argument has width * height RGBA pixels or width * height * 4 bytes
		vector<unsigned char> image;
		image.resize(width * height * 4);
		
		for(unsigned y = 0; y < height; y++)
		for(unsigned x = 0; x < width; x++)
		{
			image[4 * width * y + 4 * x + 0] = min(255u, (unsigned int)(255.*(*this)(x,y).GetPixel().r));
			image[4 * width * y + 4 * x + 1] = min(255u, (unsigned int)(255.*(*this)(x,y).GetPixel().g));
			image[4 * width * y + 4 * x + 2] = min(255u, (unsigned int)(255.*(*this)(x,y).GetPixel().b));
			image[4 * width * y + 4 * x + 3] = min(255u, (unsigned int)(255.*(*this)(x,y).GetPixel().a));
		}
		
		//Encode the image
		unsigned error = lodepng::encode(name, image, width, height);

		//if there's an error, display it
		if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
	}
};