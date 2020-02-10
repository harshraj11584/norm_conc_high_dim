import numpy as np 
from scipy.linalg import eigh
from multiprocessing import Pool 
import matplotlib.pyplot as plt


num_parrallel_proc = 3
N = int(1e3)
k = 0.8

count = 0 


def p_gaussian_kernel(distances,p=10.0,sigma=1.0):
	kernel_vals = np.exp( - ((distances/sigma)**p) ) 
	return kernel_vals 

def do_trial(dim):
	
	data = np.random.uniform(low=-1000.0,high=1000.0,size=(N,dim))
	# print(np.mean(data,axis=0).shape)
	# all_distances = 2* (np.linalg.norm(data-np.mean(data,axis=0),ord=k,axis=1)**k)
	all_distances = []
	for i in range(N):
		for j in range(i+1,N):
			all_distances.append(np.linalg.norm(data[i]-data[j],ord=k)**k)
	all_distances = np.array(all_distances)
	plt.hist(all_distances,100)
	plt.show()

	d_5, d_50, d_95 = np.min(all_distances)-1.0,np.percentile(all_distances,50),np.max(all_distances)+1.0
	
	# Uncomment one of these, alter between p-Gaussian and Gaussian Kernel
	p = (np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5)) 	
	# p = 1


	sigma = d_50/((-np.log(0.50))**(1/p))

	K_mat = np.zeros((N,N))
	for i in range(N):
		for j in range(i,N):
			if i==j:
				K_mat[i,j]=1.0
			else:
				K_mat[i,j] = p_gaussian_kernel(np.linalg.norm(data[i]-data[j],ord=k)**k,p,sigma)
				# if np.linalg.norm(data[i]-data[j],ord=k)**k<d_5 or np.linalg.norm(data[i]-data[j],ord=k)**k>d_95:
				# 	print("Found problematic data")
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
	# res = proc.map(do_trial,np.random.randint(low=5,high=1001,size=num_parrallel_proc)	)
	res = proc.map(do_trial,[38,22]	)
	proc.close() 
	if(min(res)<0):
		print("Found eigval<100, breaking")
		break




