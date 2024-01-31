#from mpi4py import MPI
#import train_framework as example
#import sys
## 调用C++函数
#print("Creating some objects:")
#c = example.Circle(10)
#print("    Created circle", c)
#s = example.Square(10)
#print("    Created square", s)
#
## ----- Access a static member -----
#
#print("\nA total of", example.cvar.Shape_nshapes, "shapes were created")
#
## ----- Member data access -----
#
## Set the location of the object
#
#c.x = 20
#c.y = 30
#
#s.x = -10
#s.y = 5
#
#print("\nHere is their current position:")
#print("    Circle = (%f, %f)" % (c.x, c.y))
#print("    Square = (%f, %f)" % (s.x, s.y))
#
## ----- Call some methods -----
#
#print("\nHere are some properties of the shapes:")
#
#for o in [c, s]:
#    print("   ", o)
#    print("        area      = ", o.area())
#    print("        perimeter = ", o.perimeter())
## prevent o from holding a reference to the last object looked at
#o = None
#
#print("\nGuess I'll clean up now")
#
## Note: this invokes the virtual destructor
#del c
#del s
#
#print(example.cvar.Shape_nshapes, "shapes remain")
#print("Goodbye")
#
#print("TEST START")
#example.main()
#print("TEST SUCCESSFUL")
## file: runme.py
#
# This file illustrates the proxy class C++ interface generated
# by SWIG.
import train_framework as ho
from mpi4py import MPI

comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# x = ho.sayhello(comm)
ho.test_main(comm)
try:
    ho.test_main(list())
except:
    pass
else:
    assert 0, "exception not raised"
# MPI.Finalize()
# print("rank is %d, x= %d"%(rank,x))
#
## ----- Object creation -----
#

