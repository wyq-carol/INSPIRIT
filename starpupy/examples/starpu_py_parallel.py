# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
import starpu
import starpu.joblib
import time
import asyncio
from math import sqrt
from math import log10
import numpy as np

#generate a list to store functions
g_func=[]

#function no input no output print hello world
def hello():
	print ("Example 1: Hello, world!")
g_func.append(starpu.joblib.delayed(hello)())

#function no input no output
def func1():
	print ("Example 2: This is a function no input no output")
g_func.append(starpu.joblib.delayed(func1)())

#function no input return a value
def func2():
	print ("Example 3:")
	return 12
g_func.append(starpu.joblib.delayed(func2)())
 
#function has 2 int inputs and 1 int output
def exp(a,b):
	res_exp=a**b
	print("Example 4: The result of ",a,"^",b,"is",res_exp)
	return res_exp
g_func.append(starpu.joblib.delayed(exp)(2, 3))

#function has 4 float inputs and 1 float output
def add(a,b,c,d):
	res_add=a+b+c+d
	print("Example 5: The result of ",a,"+",b,"+",c,"+",d,"is",res_add)
	return res_add
g_func.append(starpu.joblib.delayed(add)(1.2, 2.5, 3.6, 4.9))

#function has 2 int inputs 1 float input and 1 float output 1 int output
def sub(a,b,c):
	res_sub1=a-b-c
	res_sub2=a-b
	print ("Example 6: The result of ",a,"-",b,"-",c,"is",res_sub1,"and the result of",a,"-",b,"is",res_sub2)
	return res_sub1, res_sub2
g_func.append(starpu.joblib.delayed(sub)(6, 2, 5.9))

##########functions of array calculation###############

def scal(a, t):
	for i in range(len(t)):
		t[i]=t[i]*a
	return t

def add_scal(a, t1, t2):
	for i in range(len(t1)):
		t1[i]=t1[i]*a+t2[i]
	return t1

def scal_arr(a, t):
    for i in range(len(t)):
        t[i]=t[i]*a[i]
    return t

def multi(a,b):
	res_multi=a*b
	return res_multi

def multi_2arr(a, b):
    for i in range(len(a)):
        a[i]=a[i]*b[i]
    return a

def multi_list(l):
	res = []
	for (a,b) in l:
		res.append(a*b)
	return res

########################################################

#################scikit test###################
DEFAULT_JOBLIB_BACKEND = starpu.joblib.get_active_backend()[0].__class__
class MyBackend(DEFAULT_JOBLIB_BACKEND):  # type: ignore
    def __init__(self, *args, **kwargs):
        self.count = 0
        super().__init__(*args, **kwargs)

    def start_call(self):
        self.count += 1
        return super().start_call()


starpu.joblib.register_parallel_backend('testing', MyBackend)

with starpu.joblib.parallel_backend("testing") as (ba, n_jobs):
	print("backend and n_jobs is", ba, n_jobs)
###############################################


N=10000
# A=np.arange(N)
# B=np.arange(N)
# a=np.arange(N)
# b=np.arange(N, 2*N, 1)

#starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="log")(starpu.joblib.delayed(log10)(i+1)for i in range(N))
# for x in [10, 100, 1000, 10000, 100000, 1000000]:
# 	for X2 in range(x, x*10, x):
# 		starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="log")(starpu.joblib.delayed(log10)(i+1)for i in range(X2))
# 		print(range(X2))

print("************************")
print("parallel Normal version:")
print("************************")
print("--input is an iterable argument list, for the function which has one scalar as its argument")
start_exec1=time.time()
start_cpu1=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=3, perfmodel="sqrt")(starpu.joblib.delayed(sqrt)(i**2)for i in range(N))
end_exec1=time.time()
end_cpu1=time.process_time()
print("the program execution time is", end_exec1-start_exec1)
print("the cpu execution time is", end_cpu1-start_cpu1)

print("--inputs is an iterable argument list, for the function which has multiple scalars as its arguments ")
a=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec2=time.time()
start_cpu2=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=3, perfmodel="multi")(starpu.joblib.delayed(multi)(i,j) for i,j in zip(a,b))
end_exec2=time.time()
end_cpu2=time.process_time()
print("the program execution time is", end_exec2-start_exec2)
print("the cpu execution time is", end_cpu2-start_cpu2)

print("--inputs are iterable argument list and numpy array, for the function which has multiple arrays as its arguments")
A=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec3=time.time()
start_cpu3=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="scal_list")(starpu.joblib.delayed(scal_arr)( (i for i in b), A))
end_exec3=time.time()
end_cpu3=time.process_time()
print("the program execution time is", end_exec3-start_exec3)
print("the cpu execution time is", end_cpu3-start_cpu3)

print("--input is an iterable argument list, for the function which has one array as its argument")
a=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec4=time.time()
start_cpu4=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="multi_list")(starpu.joblib.delayed(multi_list)((i,j) for i,j in zip(a,b)))
end_exec4=time.time()
end_cpu4=time.process_time()
print("the program execution time is", end_exec4-start_exec4)
print("the cpu execution time is", end_cpu4-start_cpu4)

print("--input are multiple iterable argument lists, for the function which has multiple arrays as its arguments")
a=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec5=time.time()
start_cpu5=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="multi_2list")(starpu.joblib.delayed(multi_2arr)((i for i in a), (j for j in b)))
end_exec5=time.time()
end_cpu5=time.process_time()
print("the program execution time is", end_exec5-start_exec5)
print("the cpu execution time is", end_cpu5-start_cpu5)

print("--input are multiple numpy arrays, for the function which has multiple arrays as its arguments")
A=np.arange(N)
B=np.arange(N, 2*N, 1)
print("The input arrays are A", A, "B", B)
start_exec6=time.time()
start_cpu6=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="multi_2list")(starpu.joblib.delayed(multi_2arr)(A, B))
end_exec6=time.time()
end_cpu6=time.process_time()
print("the program execution time is", end_exec6-start_exec6)
print("the cpu execution time is", end_cpu6-start_cpu6)
print("The return arrays are A", A, "B", B)

print("--input are scalar and iterable argument list, for the function which has scalar and array as its arguments")
a=np.arange(N)
start_exec7=time.time()
start_cpu7=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="scal")(starpu.joblib.delayed(scal)(2, (j for j in a)))
end_exec7=time.time()
end_cpu7=time.process_time()
print("the program execution time is", end_exec7-start_exec7)
print("the cpu execution time is", end_cpu7-start_cpu7)

print("--input are scalar and numpy array, for the function which has scalar and array as its arguments")
A=np.arange(N)
print("The input array is", A)
start_exec8=time.time()
start_cpu8=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="scal")(starpu.joblib.delayed(scal)(2,A))
end_exec8=time.time()
end_cpu8=time.process_time()
print("the program execution time is", end_exec8-start_exec8)
print("the cpu execution time is", end_cpu8-start_cpu8)
print("The return array is", A)

print("--input are scalar and multiple numpy arrays, for the function which has scalar and arrays as its arguments")
A=np.arange(N)
B=np.arange(N)
print("The input arrays are A", A, "B", B)
start_exec9=time.time()
start_cpu9=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="add_scal")(starpu.joblib.delayed(add_scal)(2,A,B))
end_exec9=time.time()
end_cpu9=time.process_time()
print("the program execution time is", end_exec9-start_exec9)
print("the cpu execution time is", end_cpu9-start_cpu9)
print("The return arrays are A", A, "B", B)


print("--input is iterable function list")
start_exec10=time.time()
start_cpu10=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=3, perfmodel="func")(g_func)
end_exec10=time.time()
end_cpu10=time.process_time()
print("the program execution time is", end_exec10-start_exec10)
print("the cpu execution time is", end_cpu10-start_cpu10)

# def producer():
# 	for i in range(6):
# 		print('Produced %s' % i)
# 		yield i
#starpu.joblib.Parallel(n_jobs=2)(starpu.joblib.delayed(sqrt)(i) for i in producer())

print("************************")
print("parallel Future version:")
print("************************")
async def main():
	print("--input is an iterable argument list, for the function which has one scalar as its argument")
	fut1=starpu.joblib.Parallel(mode="future", n_jobs=3, perfmodel="sqrt")(starpu.joblib.delayed(sqrt)(i**2)for i in range(N))
	res1=await fut1
	#print(res1)

	print("--inputs is an iterable argument list, for the function which has multiple scalars as its arguments ")
	a=np.arange(N)
	b=np.arange(N, 2*N, 1)
	fut2=starpu.joblib.Parallel(mode="future", n_jobs=3, perfmodel="multi")(starpu.joblib.delayed(multi)(i,j) for i,j in zip(a,b))
	res2=await fut2
	#print(res2)

	print("--inputs are iterable argument list and numpy array, for the function which has multiple arrays as its arguments")
	A=np.arange(N)
	b=np.arange(N, 2*N, 1)
	fut3=starpu.joblib.Parallel(mode="future", n_jobs=3, perfmodel="scal_list")(starpu.joblib.delayed(scal_arr)( (i for i in b), A))
	res3=await fut3
	#print(res3)

	print("--input is an iterable argument list, for the function which has one array as its argument")
	a=np.arange(N)
	b=np.arange(N, 2*N, 1)
	fut4=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="multi_list")(starpu.joblib.delayed(multi_list)((i,j) for i,j in zip(a,b)))
	res4=await fut4
	#print(res4)

	print("--input are multiple iterable argument lists, for the function which has multiple arrays as its arguments")
	a=np.arange(N)
	b=np.arange(N, 2*N, 1)
	fut5=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="multi_2list")(starpu.joblib.delayed(multi_2arr)((i for i in a), (j for j in b)))
	res5=await fut5
	#print(res5)

	print("--input are multiple numpy arrays, for the function which has multiple arrays as its arguments")
	A=np.arange(N)
	B=np.arange(N, 2*N, 1)
	print("The input arrays are A", A, "B", B)
	fut6=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="multi_2list")(starpu.joblib.delayed(multi_2arr)(A, B))
	res6=await fut6
	print("The return arrays are A", A, "B", B)


	print("--input are scalar and iterable argument list, for the function which has scalar and array as its arguments")
	a=np.arange(N)
	fut7=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="scal")(starpu.joblib.delayed(scal)(2, (j for j in a)))
	res7=await fut7
	#print(res6)

	print("--input are scalar and numpy array, for the function which has scalar and array as its arguments")
	A=np.arange(N)
	print("The input array is", A)
	fut8=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="scal")(starpu.joblib.delayed(scal)(2,A))
	res8=await fut8
	print("The return array is", A)

	print("--input are scalar and multiple numpy arrays, for the function which has scalar and arrays as its arguments")
	A=np.arange(N)
	B=np.arange(N)
	print("The input arrays are A", A, "B", B)
	fut9=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="add_scal")(starpu.joblib.delayed(add_scal)(2,A,B))
	res9=await fut9
	print("The return arrays are A", A, "B", B)

	print("--input is iterable function list")
	fut10=starpu.joblib.Parallel(mode="future", n_jobs=2)(g_func)
	res10=await fut10
	#print(res9)

asyncio.run(main())

starpu.perfmodel_plot(perfmodel="sqrt",view=False)
starpu.perfmodel_plot(perfmodel="multi",view=False)
starpu.perfmodel_plot(perfmodel="func",view=False)
