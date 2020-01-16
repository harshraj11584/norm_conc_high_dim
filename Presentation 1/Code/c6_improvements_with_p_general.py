import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from math import pi,e
from kernel_implementations import * 

dimensions = [ 2, 10, 100 ]
N = 10000

df1 = lambda x : get_distances(x,order=2.0) 


# for dim in dimensions:
# 	sigma_data=1.0
# 	actual_data = np.random.multivariate_normal(mean=np.zeros(dim), cov=sigma_data**2 * np.eye(dim), size=N) 
# 	actual_distances = df1(actual_data) 
# 	d_5, d_50, d_95 = np.percentile(actual_distances,5),np.percentile(actual_distances,50),np.percentile(actual_distances,95)
# 	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 

# 	dist_vals = np.linspace( 0, np.max(actual_distances)+5 , 100)
# 	kernel_vals = p_polynomial_kernel(dist_vals,p=p) 
# 	kernel_vals = kernel_vals/np.max(kernel_vals)

# 	fig,ax=plt.subplots()
# 	ax.plot(dist_vals,kernel_vals,label="Kernel Values") 
# 	ax.hist(actual_distances,bins=200,density=True,label="Distribution of Distances") 
# 	ax.axvline(d_5,ls='--')
# 	ax.axvline(d_95,ls='--')
# 	ax.set_title("p-Polynomial (Dim="+str(dim)+")") 
# 	ax.legend() 
# 	# fig.savefig('../Graphs/g5_problem_polynomial_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.uniform(low=0.0,high=1.0,size=(N,dim))  
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
	# ax.set_ylabel('Kernel Values')
	ax.set_ylim(0,2.0)
	ax.set_title("p-Rational Quadratic (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g6_improved_p_RationalQuadratic_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.uniform(low=0.0,high=1.0,size=(N,dim))  
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
	# ax.set_ylabel('Kernel Values')
	ax.set_ylim(0,2.0)
	ax.set_title("p-Inverse Multi-Quadratic (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g6_improved_p_InverseMultiQuadratic_dim='+str(dim)+'.png')



for dim in dimensions:
	sigma_data=1.0
	actual_data =np.random.uniform(low=0.0,high=1.0,size=(N,dim))  
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
	# ax.set_ylabel('Kernel Values')
	ax.set_ylim(0,2.0)
	ax.set_title("p-Cauchy (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g6_improved_p_Cauchy_dim='+str(dim)+'.png')


for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.uniform(low=0.0,high=1.0,size=(N,dim)) 
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
	# ax.set_ylabel('Kernel Values')
	ax.set_ylim(0,2.0)
	ax.set_title("p-Matern Kernel (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g6_improved_p_Matern_dim='+str(dim)+'.png')

for dim in dimensions:
	sigma_data=1.0
	actual_data = np.random.uniform(low=0.0,high=1.0,size=(N,dim))  
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
	# ax.set_ylabel('Kernel Values')
	ax.set_ylim(0,2.0)
	ax.set_title("p-LaPlace Kernel (d="+str(dim)+")")
	ax.legend() 
	fig.savefig('../Graphs/g6_improved_p_LaPlace_dim='+str(dim)+'.png')


# plt.show()


