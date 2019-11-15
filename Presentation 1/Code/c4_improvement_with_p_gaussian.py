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
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 

	#estimating hyperparameters p, sigma using thumb rule in paper 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
	sigma = d_50/((-np.log(0.50))**(1/p))
	sigma1 = d_5/((-np.log(0.05))**(1/p))
	sigma2 = d_95/((-np.log(0.95))**(1/p))


	dist_vals = np.linspace( 0, np.max( (np.max(actual_distances)+5, 15) ) , 100)
	kernel_vals = p_gaussian_kernel(dist_vals,p,sigma) 
	kernel_vals1 = p_gaussian_kernel(dist_vals,p,sigma1) 
	kernel_vals2 = p_gaussian_kernel(dist_vals,p,sigma2) 


	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals1,label="Kernel Values (sigma="+str(sigma1)[:4]+")") 
	ax.plot(dist_vals,kernel_vals,label="Kernel Values (sigma="+str(sigma)[:4]+")") 
	ax.plot(dist_vals,kernel_vals2,label="Kernel Values (sigma="+str(sigma2)[:4]+")") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_title("p-Gaussian (Dimension="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g4_improvement_with_p_Gaussian_dim='+str(dim)+'.png')

# plt.show()


