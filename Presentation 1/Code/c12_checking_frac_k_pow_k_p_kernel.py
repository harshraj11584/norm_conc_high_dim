import numpy as np 
import itertools
from scipy.linalg import eigh
from multiprocessing import Pool 


num_trials = 4
N = int(1e3)
k = 2.0

count = 0 

def p_gaussian_kernel(distances,p=10.0,sigma=1.0):
	kernel_vals = np.exp( - (distances**p)/(sigma**p) ) 
	return kernel_vals 

def do_trial(dim):
	
	data = np.random.uniform(low=-1000.0,high=1000.0,size=(N,dim))
	# print("Trial Num= ",i+1,"Dimension=",dim)

	dist_mat = list(itertools.product(np.arange(N),repeat=2))
	# print(dist_mat)
	elem_mat = data[dist_mat]
	# print(elem_mat)
	# print(elem_mat.shape)

	diff_mat = elem_mat[:,0,:] - elem_mat[:,1,:]
	# print(diff_mat.shape)
	norm_vals = np.linalg.norm(diff_mat,axis=1,ord=k)
	# print(norm_vals.shape)
	d_5, d_50, d_95 = np.percentile(norm_vals,5),np.percentile(norm_vals,50),np.percentile(norm_vals,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
	sigma = d_50/((-np.log(0.50))**(1/p))
	kernel_vals = p_gaussian_kernel(norm_vals,p,sigma) 
	K_mat = kernel_vals.reshape((N,N))

	lowest_eig_val = eigh(K_mat,eigvals_only=True,eigvals=(0,0),overwrite_a=True)[0]
	global count 
	count = count+1
	print("Trial Num= ",count,"Dimension=",dim,"Lowest EigVal = ",lowest_eig_val)
	return lowest_eig_val




while(True and count < 100):
	proc = Pool(num_trials)
	res = proc.map(do_trial,np.random.randint(low=5,high=1001,size=num_trials)	)
	proc.close() 
	if(min(res)<0):
		print("Found eigval<100, breaking")
		break




