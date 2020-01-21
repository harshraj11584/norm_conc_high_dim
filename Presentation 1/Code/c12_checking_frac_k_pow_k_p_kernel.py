import numpy as np 
from scipy.linalg import eigh
from multiprocessing import Pool 


num_parrallel_proc = 8
N = int(1e3)
k = 2.0

count = 0 


def p_gaussian_kernel(distances,p=10.0,sigma=1.0):
	kernel_vals = np.exp( - ((distances/sigma)**p) ) 
	return kernel_vals 

def do_trial(dim):
	
	data = np.random.uniform(low=-1000.0,high=1000.0,size=(N,dim))
	# print(np.mean(data,axis=0).shape)
	all_distances = np.linalg.norm(data-np.mean(data,axis=0),ord=k,axis=1)

	d_5, d_50, d_95 = np.percentile(all_distances,5),np.percentile(all_distances,50),np.percentile(all_distances,95)
	
	# Uncomment one of these, alter between p-Gaussian and Gaussian Kernel
	p = int(np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5)) 	
	# p = 1


	sigma = d_50/((-np.log(0.50))**(1/p))

	K_mat = np.zeros((N,N))
	for i in range(N):
		for j in range(i,N):
			K_mat[i,j] = p_gaussian_kernel(np.linalg.norm(data[i]-data[j],ord=k),p,sigma)
			K_mat[j,i] = K_mat[i,j]

	lowest_eig_val = eigh(K_mat,eigvals_only=True,eigvals=(0,0))[0]
	global count 
	count = count+1
	print("Tried Dim=",dim,"p=",p,"Lowest EigVal = ",lowest_eig_val)
	return lowest_eig_val

# N = 10
# do_trial(100)

while(True and count < 100):
	proc = Pool(num_parrallel_proc)
	res = proc.map(do_trial,np.random.randint(low=5,high=1001,size=num_parrallel_proc)	)
	proc.close() 
	if(min(res)<0):
		print("Found eigval<100, breaking")
		break




