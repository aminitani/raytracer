#Amin Itani
#raytracer makefile

raytracer: raytracer.o
	g++ -o raytracer raytracer.o

raytracer.o: ./raytracer.cpp ./RayMath.h
	g++ -Wall -c -std=c++11 -pthread raytracer.cpp

clean:
	rm -f raytracer && rm -f *.o
