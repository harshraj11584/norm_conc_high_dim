import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import pi,e
from kernel_implementations import * 

dimensions = [ 5, 10, 100 ]
N = 10000

k = 0.6
df1 = lambda x : get_distances(x,order=k) 

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal( mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size = N ) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
	sigma = d_50/((-np.log(0.5))**0.5)


	dist_vals = np.linspace( 0, np.max( (np.max(actual_distances)+5, 15) ) , 100)
	kernel_vals = gaussian_kernel(dist_vals,sigma) 
	kernel_vals = kernel_vals/np.max(kernel_vals)


	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$Gaussian (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_Gaussian_dim='+str(dim)+'.png')

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = polynomial_kernel(dist_vals) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$Polynomial (d="+str(dim)+")") 
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_polynomial_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = rational_quadratic_kernel(dist_vals,c=(d_50)**2) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$Rational Quadratic (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_RationalQuadratic_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = inverse_multiquadratic_kernel(dist_vals,c=(d_50)/(3**0.5)) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$Inverse Multi-Quadratic (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_InverseMultiQuadratic_dim='+str(dim)+'.png')



for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = cauchy_kernel(dist_vals,s=d_50) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$Cauchy (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_Cauchy_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = matern_kernel(dist_vals,l=d_50) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$Matern Kernel (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_Matern_dim='+str(dim)+'.png')

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
	actual_distances = df1(actual_data) 
	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)

	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
	kernel_vals = laplace_kernel(dist_vals,sigma=d_50/(-np.log(0.5))) 
	kernel_vals = kernel_vals/np.max(kernel_vals)

	fig,ax=plt.subplots()
	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
	ax.hist(actual_distances,bins=50,density=True,label="Distribution of Distances") 
	ax.axvline(d_5,ls='--')
	ax.axvline(d_95,ls='--')
	ax.set_xlabel('Distance')
	ax.set_ylabel('Kernel Values')
	ax.set_title("$L_{"+str(k)[:4]+"}$LaPlace Kernel (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g7_problem_L('+ str(k) +')_LaPlace_dim='+str(dim)+'.png')


# plt.show()


