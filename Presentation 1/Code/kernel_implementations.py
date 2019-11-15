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






