#Amin Itani
#raytracer makefile

raytracer: main.o lodepng.o
	g++ -o raytracer main.o lodepng.o && rm -f *.o

main.o: ./main.cpp
	g++ -Wall -c -std=c++11 -pthread main.cpp

lodepng.o: ./png/lodepng.cpp
	g++ -Wall -c -std=c++11 -pthread ./png/lodepng.cpp

clean:
	rm -f raytracer && rm -f *.o
