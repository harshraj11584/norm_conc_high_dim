import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import pi,e


def get_distances(data, order=2):
	# assuming data is Nxd, N points in d dimensions
	dists = (np.sum(abs(data)**order,axis=1))**(1/order) 
	return dists 

def gaussian_kernel(distances,sigma=1.0): 
	kernel_vals = np.exp( - (distances**2)/(sigma**2) ) 
	return kernel_vals 

def p_gaussian_kernel(distances,p=10.0,sigma=1.0):
	kernel_vals = np.exp( - (distances**p)/(sigma**p) ) 
	return kernel_vals 

def polynomial_kernel(distances,a=1,b=0,p=2):
	a = 1/len(distances)
	kernel_vals = (a*distances+b)**p 
	kernel_vals = -kernel_vals + np.max(kernel_vals)
	return kernel_vals
# Problem : Must decrease with distance 

def rational_quadratic_kernel(distances,c=1):
	kernel_vals = 1 - (distances**2)/((distances**2)+c)
	return kernel_vals 

def inverse_multiquadratic_kernel(distances,c=1):
	kernel_vals = 1/ (( distances**2 + c**2 )**0.5)
	return kernel_vals

def cauchy_kernel(distances,s=1):
	kernel_vals = 1 / (1 + (distances**2 / s**2))
	return kernel_vals 

def matern_kernel(distances,l=1.0):
	kernel_vals = (1 + (5**0.5)*distances/l + 5*(distances**2)/(3*l**2))*np.exp( -(5**0.5)*distances/l ) 
	return kernel_vals 

def laplace_kernel(distances,sigma=1.0):
	kernel_vals = np.exp( - (distances)/(sigma) ) 
	return kernel_vals 




