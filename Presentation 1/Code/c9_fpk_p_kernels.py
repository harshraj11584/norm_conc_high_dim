import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import pi,e
from kernel_implementations import * 

dimensions = [ 2, 10, 100 ]
N = 10000 

k = 0.6
df1 = lambda x : get_distances(x,order=k)**k 

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
	sigma = d_50/((-np.log(0.50))**(1/p))
	
	dist_vals = np.linspace( 0, np.max( (np.max(actual_distances)+5, 15) ) , 100)
	kernel_vals = p_gaussian_kernel(dist_vals,p,sigma) 

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("fractional^p p-Gaussian (Dimension d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g9_improvement_fp_Gaussian_dim='+str(dim)+'.png')

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = p_rational_quadratic_kernel(dist_vals,c=(d_50)**2,p=p) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("fractional^p p-Rational Quadratic (Dimension d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g9_improved_fp_RationalQuadratic_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = p_inverse_multiquadratic_kernel(dist_vals,c=(d_50)/(3**(1/p)),p=p) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("fractional^p p-Inverse Multi-Quadratic (Dimension d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g9_improved_fp_InverseMultiQuadratic_dim='+str(dim)+'.png')



for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = p_cauchy_kernel(dist_vals,s=d_50,p=p) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("fractional^p p-Cauchy (Dimension d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g9_improved_fp_Cauchy_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = p_matern_kernel(dist_vals,l=d_50,p=p) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("fractional^p p-Matern Kernel (Dimension d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g9_improved_fp_Matern_dim='+str(dim)+'.png')

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = p_laplace_kernel(dist_vals,sigma=d_50/((-np.log(0.5))**(1/p)),p=p) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("fractional^p p-LaPlace Kernel (Dimension d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g9_improved_fp_LaPlace_dim='+str(dim)+'.png')


# plt.show()


