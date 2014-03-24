#include <iostream>
#include <vector>
#include "./RayMath.h"
#include "./png/lodepng.h"

using std::cout; using std::endl;
using std::vector;

struct pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

class Image
{
private:
	unsigned int width;
	unsigned int height;
	std::vector<pixel> pixels;
	
public:
	Image(unsigned int inWidth, unsigned int inHeight)
	{
		width = inWidth; height = inHeight;
		pixels.resize(width * height);
	}
	
	unsigned int Width() {return width;}
	unsigned int Height() {return height;}
	pixel& operator()(unsigned int x, unsigned int y) { return pixels[width * y + x]; }
};

void savePNG(const char* filename, Image &rawImage)
{
	//generate some image
	//The image argument has width * height RGBA pixels or width * height * 4 bytes
	unsigned width = rawImage.Width(), height = rawImage.Height();
	vector<unsigned char> image;
	image.resize(width * height * 4);
	
	for(unsigned y = 0; y < height; y++)
	for(unsigned x = 0; x < width; x++)
	{
		image[4 * width * y + 4 * x + 0] = rawImage(x,y).r;
		image[4 * width * y + 4 * x + 1] = rawImage(x,y).g;
		image[4 * width * y + 4 * x + 2] = rawImage(x,y).b;
		image[4 * width * y + 4 * x + 3] = rawImage(x,y).a;
	}
	
	//Encode the image
	unsigned error = lodepng::encode(filename, image, width, height);

	//if there's an error, display it
	if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}


int main()
{
	std::cout << "test" << std::endl;
	std::cout << "value of testInt is: " << RayMath::testInt << std::endl;
	
	Image image(512, 512);
	
	//trace some rays///////////////////////////////////////
	
	for(unsigned y = 0; y < image.Height()/2; y++)
	for(unsigned x = 0; x < image.Width()/2; x++)
	{
		image(x,y).r = 255;
		image(x,y).g = 0;
		image(x,y).b = 255;
		image(x,y).a = 255;
	}
	for(unsigned y = 0; y < image.Height()/2; y++)
	for(unsigned x = image.Width()/2; x < image.Width(); x++)
	{
		image(x,y).r = 0;
		image(x,y).g = 255;
		image(x,y).b = 0;
		image(x,y).a = 127;
	}
	for(unsigned y = image.Height()/2; y < image.Height(); y++)
	for(unsigned x = 0; x < image.Width()/2; x++)
	{
		image(x,y).r = 255;
		image(x,y).g = 255;
		image(x,y).b = 255;
		image(x,y).a = 0;
	}
	for(unsigned y = image.Height()/2; y < image.Height(); y++)
	for(unsigned x = image.Width()/2; x < image.Width(); x++)
	{
		image(x,y).r = 0;
		image(x,y).g = 0;
		image(x,y).b = 0;
		image(x,y).a = 255;
	}
	
	//trace some rays///////////////////////////////////////
	
	savePNG("test.png", image);
	
	return 0;
}
