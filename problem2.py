import numpy as np


def k(x,theta):
	return np.piecewise(x, [x<0.25, np.logical_and(x>=0.25, x<0.5), np.logical_and(x<0.75, x>= 0.5), (x>=0.75)], theta)

def finite_diff(M, theta = [0.5,1,1.5,2], f = lambda x: 100*x):
	
	h = 1/M
	x = np.arange(0,1, h)

	sides = np.ones(M-1)
	mid = np.ones(M)*-2
	A = np.diag(sides,-1) + np.diag(sides,1) + np.diag(mid)
	A[0] = np.zeros(M)
	A[M-1] = np.zeros(M) 
	A[0,0] = 1
	A[-1,-1] = 1
	b = -f(x)*h**2/k(x,theta)
	b[0] = 1
	b[M-1] = 5

	return np.linalg.solve(A,b)

