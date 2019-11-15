import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import pi,e
from kernel_implementations import * 

dimensions = [ 2, 5, 10, 100 ]
N = 10000

df1 = lambda x : get_distances(x,order=2.0) 


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal( mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size = N ) 
	actual_distances = df1(actual_data) 

	#estimating hyperparameter sigma using thumb rule in paper 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	sigma = d_50/((-np.log(0.5))**0.5)
	sigma1 = d_5/((-np.log(0.05))**0.5)
	sigma2 = d_95/((-np.log(0.95))**0.5)


	dist_vals = np.linspace( 0, np.max( (np.max(actual_distances)+5, 15) ) , 100)
	kernel_vals = gaussian_kernel(dist_vals,sigma) 
	kernel_vals1 = gaussian_kernel(dist_vals,sigma1) 
	kernel_vals2 = gaussian_kernel(dist_vals,sigma2) 


	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals1,label="Kernel Values (sigma="+str(sigma1)[:4]+")") 
	ax.plot(dist_vals,kernel_vals,label="Kernel Values (sigma="+str(sigma)[:4]+")") 
	ax.plot(dist_vals,kernel_vals2,label="Kernel Values (sigma="+str(sigma2)[:4]+")") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_title("Dimension="+str(dim))
	ax.legend() 
	fig.savefig('../Graphs/g3_problem_with_Gaussian_dim='+str(dim)+'.png')

# plt.show()


