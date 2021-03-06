import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import pi,e
from kernel_implementations import * 

dimensions = [ 2, 10, 50, 100 ]
N = 1000

df1 = lambda x : get_distances(x,order=2.0) 


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.uniform(low=0.0, high=1.0, size=(N,dim)) 
	actual_distances = df1(actual_data) 

	#estimating hyperparameter sigma using thumb rule in paper 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	sigma = d_50/((-np.log(0.5))**0.5)
	sigma1 = d_5/((-np.log(0.05))**0.5)
	sigma2 = d_95/((-np.log(0.95))**0.5)


	# dist_vals = np.linspace( 0, np.max( (np.max(actual_distances)+5, 15) ) , 100)
	kernel_vals = gaussian_kernel(actual_distances,sigma) 
	# kernel_vals1 = gaussian_kernel(dist_vals,sigma1) 
	# kernel_vals2 = gaussian_kernel(dist_vals,sigma2) 

	mean, stddev = np.mean(actual_distances), np.std(actual_distances,ddof=1)
	a,b = gaussian_kernel(np.array([mean-3*stddev,mean+3*stddev]),sigma)



	fig,ax=plt.subplots(figsize=(10,6))
	# ax.plot(dist_vals,kernel_vals1,label="Kernel Values ($\sigma$="+str(sigma1)[:4]+")") 
	# ax.plot(dist_vals,kernel_vals,label="Kernel Values ($\sigma$="+str(sigma)[:4]+")") 
	# ax.plot(dist_vals,kernel_vals2,label="Kernel Values ($\sigma$="+str(sigma2)[:4]+")") 
	ax.hist(kernel_vals,bins=100,density=True) 
	ax.axvline(a,ls='--',color='r')
	if dim==2:
		ax.text(a-.20,13,'$K(\mu-3\sigma)=$'+str(a)[:4])
	else:
		ax.text(a,13,'$K(\mu-3\sigma)=$'+str(a)[:4])
	ax.axvline(b,ls='--',color='r')
	ax.text(b,13,'$K(\mu+3\sigma)=$'+str(b)[:4])
	ax.set_title("Histogram of Kernel Values (Dimension D="+str(dim)+")")
	ax.set_xlabel('Kernel Values')
	ax.set_ylabel('Frequency')
	ax.set_ylim(0.0,15)
	ax.set_xlim(0,1)
	ax.legend() 
	fig.savefig('../Graphs/g11_histogram_of_kernel_values_dim='+str(dim)+'.png')

	# plt.show()


