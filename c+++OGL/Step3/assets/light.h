#pragma once
#include "../math/vec3.h"

class Light
{
	public:
		Vec3 position;
		//TODO: replace brightness with color
		float brightness;
		
		Light(Vec3 pos, float bright)
		{
			position = pos;
			brightness = bright;
		}
		
		// Vec3 &Position() {return position;}
		// float &Brightness() {return brightness;}
};