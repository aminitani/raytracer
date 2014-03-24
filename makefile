#Amin Itani
#raytracer makefile

raytracer: raytracer.o lodepng.o
	g++ -o raytracer raytracer.o lodepng.o && rm -f *.o

raytracer.o: ./raytracer.cpp ./RayMath.h
	g++ -Wall -c -std=c++11 -pthread raytracer.cpp

lodepng.o: ./png/lodepng.cpp ./png/lodepng.h
	g++ -Wall -c -std=c++11 -pthread ./png/lodepng.cpp

clean:
	rm -f raytracer && rm -f *.o
