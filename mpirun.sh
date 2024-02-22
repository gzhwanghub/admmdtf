#!/bin/bash
if [ ! -d "build"]; then
	mkdir build
else
	rm -rf build/*
fi
cd build
cmake ..
make
cp ../try.py .
mpirun -np 17 -hostfile ../hostfile python try.py
#mpirun -np 16 -hostfile ../hostfile python try.py


