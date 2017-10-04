all:
	g++ -O2 -I../include simulate.cpp -std=c++11 ../bin/libmujoco131.so ../bin/libglfw.so.3 -o ../bin/simulate
	g++ -O2 -I../include test.cpp -std=c++11 ../bin/libmujoco131.so -o ../bin/test
	g++ -O2 -I../include compile.cpp -std=c++11 ../bin/libmujoco131.so -o ../bin/compile
