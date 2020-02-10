import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm 

import sklearn.datasets
import sklearn.model_selection
from sklearn.svm import SVC
import sklearn.metrics
from sklearn.preprocessing import StandardScaler

from numba import autojit,prange
from multiprocessing import Pool 


def prep_grids(X, kd):
	ranges = []
	for i in range(X.shape[1]):
		this_dim_vals = X[:,i].reshape((X.shape[0],))
		this_dim_vals = np.sort(this_dim_vals)
		splitted_values = np.array_split(this_dim_vals,kd)
		intervals = [] 
		for i in range(len(splitted_values)):
			intervals.append( (splitted_values[i][0],splitted_values[i][-1]) )
		ranges.append(intervals)
	return np.array(ranges) # (num_dim x kd x 2)

# X = np.random.uniform(low=0,high=5,size=(6,4))
# print(X)
# grid = prep_grids(X,3)
# print(res)
# print(res.shape)

@autojit
def get_distance(x,y,grid,p=1.0):
	dim = grid.shape[0]
	x = x.reshape((dim,))
	y = y.reshape((dim,))
	kd = grid.shape[1]
	c = (np.abs(x-y))
	b = np.zeros((dim,))
	a = np.zeros((dim,))
	for d in prange(dim):
		xd = x[d]
		yd = y[d]
		ind_xd = np.searchsorted(grid[d,:,0],xd)-1
		if grid[d,ind_xd,0]<=xd<=grid[d,ind_xd,1]:
			ind_xd = ind_xd
		elif ind_xd>=1 and grid[d,ind_xd-1,0]<=xd<=grid[d,ind_xd-1,1]:
			ind_xd = ind_xd -1
		elif ind_xd <= grid.shape[1]-2 and grid[d,ind_xd+1,0]<=yd<=grid[d,ind_xd+1,1]:
			ind_xd = ind_xd+1
		# print(grid[d,ind_xd-1,0]<=xd<=grid[d,ind_xd-1,1])
		if grid[d,ind_xd,0]<=yd<=grid[d,ind_xd,1] and grid[d,ind_xd,0]!=grid[d,ind_xd,1]:
			b[d] = (1./(grid[d,ind_xd,1]-grid[d,ind_xd,0]))
			a[d] = 1.0
			# print(b[d])
		# for i in range(kd):
		# 	if grid[d,i,0]<=xd<=grid[d,i,1]:
		# 		if grid[d,i,0]<=yd<=grid[d,i,1]:
		# 			b[d] = (1./(grid[d,i,0]-grid[d,i,1]))
		# 		break
	
	dist = (np.sum((a - b*c)**p))**(1./p)
	# dist = dist/((np.sum(np.ones(dim)**p))**(1./p))
	# print(dist)

	if np.isnan(dist):
		print(np.sum(a-b*c))
		print(dist)
	return dist

def prep_data(n_samples=500,n_features=10,n_classes=2):
	X,Y = sklearn.datasets.make_classification(n_samples,n_features,n_informative=int(0.8*n_features),n_classes=n_classes,n_clusters_per_class=8,random_state=42)
	x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.3,random_state=42)
	return x_train,x_test,y_train,y_test

def get_new_kernel_matrix(x1,x2,grid,p=1.0):
	kernel_mat = np.zeros((len(x1),len(x2)))
	all_distances = []
	for i in tqdm(range(len(x1)),desc="new-ker"):
		for j in range(len(x2)):
			d = get_distance(x1[i],x2[j],grid,p)
			all_distances.append(d)
			# kernel_mat[i,j] = np.exp(-(d**2))
			kernel_mat[i,j] = d
	# all_distances = all_distances.reshape((len(x1)*len(x2),))
	# sigma = -(np.percentile(np.array(all_distances),50)**2)/np.log(0.5)
	# fig,ax=plt.subplots()
	# plt.hist(np.array(all_distances),100,label="hist of dist")
	# plt.legend()
	# plt.savefig('hist_of_dist_new_'+str(sigma)+'.png')
	# kernel_mat = kernel_mat**(1./sigma)
	return kernel_mat

def test_new_kernel(x_train,x_test,y_train,y_test,kd):
	clf = sklearn.svm.SVC(C=regularization_val,kernel='precomputed',cache_size=1024*4)
	print("Computing Train Kernel Mat ...")
	grid = prep_grids(x_train,kd)
	kernel_train = get_new_kernel_matrix(x_train,x_train,grid,p=1.0)
	print("Training ...")
	clf.fit(kernel_train, y_train)
	y_learnt = clf.predict(kernel_train)
	score_tr = sklearn.metrics.accuracy_score(y_true=y_train,y_pred=y_learnt)
	print("Computing Test Kernel Mat ...")
	kernel_test = get_new_kernel_matrix(x_test,x_train,grid,p=1.0)
	print("Testing ...")
	y_pred = clf.predict(kernel_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	return score_tr,score


def test_rbf_kernel(x_train,x_test,y_train,y_test):
	print("Training RBF...")
	clf = sklearn.svm.SVC(C=regularization_val,gamma='scale',kernel='rbf',cache_size=1024*4)
	clf.fit(x_train,y_train)
	# print("Testing ...")
	y_learnt = clf.predict(x_train)
	score_tr = sklearn.metrics.accuracy_score(y_true=y_train,y_pred=y_learnt)
	y_pred = clf.predict(x_test) 
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	return score_tr,score 


def experiment(dim):
	x_train,x_test,y_train,y_test = prep_data(n_samples=1000,n_features=dim)
	rbf_score = test_rbf_kernel(x_train,x_test,y_train,y_test)
	print("dim=",dim,"RBF Scores = ", rbf_score)

	kd = int(np.ceil(theta*dim))
	new_score = test_new_kernel(x_train,x_test,y_train,y_test,kd)
	print("dim=",dim,"New Scores = ", new_score)
	print("Dim=",dim,"Done")
	return (rbf_score,new_score)


p=1.0
theta = 1.0
regularization_val= 1000.0
experiment(300)
