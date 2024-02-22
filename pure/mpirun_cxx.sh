#!/bin/bash
cd ..
if [ ! -d "build"]; then
	mkdir build
else
	rm -rf build/*
fi
cd build
cmake ../pure/
make
mpirun -np 16 -hostfile ../hostfile ./train_framework


