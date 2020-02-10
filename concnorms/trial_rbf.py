import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

from tqdm import tqdm 

import sklearn.datasets
import sklearn.model_selection
from sklearn.svm import SVC
import sklearn.metrics
from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool 

def prep_data(n_samples=500,n_features=10,n_classes=2):
	X,Y = sklearn.datasets.make_classification(n_samples,n_features,n_informative=int(0.8*n_features),n_classes=n_classes,n_clusters_per_class=2,random_state=42)
	# X = StandardScaler().fit_transform(X)
	# print(X.shape,Y.shape)
	x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.3,random_state=42)
	return x_train,x_test,y_train,y_test

def test_rbf_kernel(x_train,x_test,y_train,y_test):
	# print("Training ...")
	clf = sklearn.svm.SVC(C=regularization_val,gamma='scale',kernel='rbf',cache_size=1024*4)
	clf.fit(x_train,y_train)
	# print("Testing ...")
	y_learnt = clf.predict(x_train)
	score_tr = sklearn.metrics.accuracy_score(y_true=y_train,y_pred=y_learnt)
	y_pred = clf.predict(x_test) 
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	return score_tr,score 

def p_gaussian_kernel(distances,p=10.0,sigma=1.0):
	kernel_vals = np.exp( - (distances/sigma)**p ) 
	return kernel_vals 

def get_p_kernel_matrix(x1,x2,k,p,sigma):
	df1 = lambda x: ((np.sum(np.abs(x)**k,axis=1))**(1/k) if len(x.shape)>1 else (np.sum(np.abs(x)**k))**(1/k))
	kernel_mat = np.zeros((len(x1),len(x2)))
	for i in tqdm(range(len(x1)),desc="p-Ker"):
		for j in range(len(x2)):
			kernel_mat[i,j] = p_gaussian_kernel(df1(x1[i]-x2[j]),p,sigma)
	return kernel_mat

def get_fp_kernel_matrix(x1,x2,k,p,sigma):
	df1 = lambda x: ((np.sum(np.abs(x)**k,axis=1)) if len(x.shape)>1 else (np.sum(np.abs(x)**k)))
	kernel_mat = np.zeros((len(x1),len(x2)))
	for i in tqdm(range(len(x1)),desc="fp-ker"):
		for j in range(len(x2)):
			kernel_mat[i,j] = p_gaussian_kernel(df1(x1[i]-x2[j]),p,sigma)
	return kernel_mat

# def get_fpkp_kernel_matrix(x1,x2,k,p,sigma):
# 	df1 = lambda x: ((np.sum(abs(x)**k,axis=1)) if len(x.shape)>1 else (np.sum(abs(x)**k)))
# 	kernel_mat = np.zeros((len(x1),len(x2)))
# 	for i in range(len(x1)):
# 		for j in range(len(x2)):
# 			kernel_mat[i,j] = p_gaussian_kernel(df1(x1[i]-x2[j]),p,sigma)
# 	return kernel_mat

def test_p_gaussian_kernel(x_train,x_test,y_train,y_test):
	clf = sklearn.svm.SVC(C=regularization_val,kernel='precomputed',cache_size=1024*4)
	# print("Computing Train Kernel Mat ...")
	k = 2.0	
	df1 = lambda x: ((np.sum(abs(x)**k,axis=1))**(1/k) if len(x.shape)>1 else (np.sum(abs(x)**k))**(1/k))
	# all_dists = df1(x_train-np.mean(x_train,axis=0))
	all_dists = []
	for i in range(len(x_train)):
		for j in range(i+1,len(x_train)):
			all_dists.append(df1(x_train[i]-x_train[j]))
	all_dists = np.array(all_dists)
	d_5, d_50, d_95 = np.min(all_dists),np.percentile(all_dists,50),np.max(all_dists)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
	sigma = d_50/((-np.log(0.50))**(1/p))
	kernel_train = get_p_kernel_matrix(x_train,x_train,k,p,sigma)
	# print("Training ...")
	clf.fit(kernel_train, y_train)
	y_learnt = clf.predict(kernel_train)
	score_tr = sklearn.metrics.accuracy_score(y_true=y_train,y_pred=y_learnt)
	# print("Computing Test Kernel Mat ...")
	kernel_test = get_p_kernel_matrix(x_test,x_train,k,p,sigma)
	# print("Testing ...")
	y_pred = clf.predict(kernel_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	return score_tr,score

def test_fp_gaussian_kernel(x_train,x_test,y_train,y_test):
	clf = sklearn.svm.SVC(C=regularization_val,kernel='precomputed',cache_size=1024*4)
	# print("Computing Train Kernel Mat ...")
	k = 0.5
	df1 = lambda x: ((np.sum(abs(x)**k,axis=1)) if len(x.shape)>1 else (np.sum(abs(x)**k)))
	# all_dists = df1(x_train-np.mean(x_train,axis=0))
	all_dists = []
	for i in range(len(x_train)):
		for j in range(i+1,len(x_train)):
			all_dists.append(df1(x_train[i]-x_train[j]))
	all_dists = np.array(all_dists)
	d_5, d_50, d_95 = np.min(all_dists),np.percentile(all_dists,50),np.max(all_dists)
	d_5, d_50, d_95 = np.min(all_dists),np.percentile(all_dists,50),np.max(all_dists)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
	sigma = d_50/((-np.log(0.50))**(1/p))
	kernel_train = get_fp_kernel_matrix(x_train,x_train,k,p,sigma)
	# print("Training ...")
	clf.fit(kernel_train, y_train)
	y_learnt = clf.predict(kernel_train)
	score_tr = sklearn.metrics.accuracy_score(y_true=y_train,y_pred=y_learnt)
	# print("Computing Test Kernel Mat ...")
	kernel_test = get_fp_kernel_matrix(x_test,x_train,k,p,sigma)
	# print("Testing ...")
	y_pred = clf.predict(kernel_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	return score_tr,score

# def test_fpkp_gaussian_kernel(x_train,x_test,y_train,y_test):
# 	clf = sklearn.svm.SVC(C=regularization_val,kernel='precomputed',cache_size=200)
# 	print("Computing Train Kernel Mat ...")
# 	k = 0.5
# 	df1 = lambda x: ((np.sum(abs(x)**k,axis=1))if len(x.shape)>1 else (np.sum(abs(x)**k)))
# 	all_dists = df1(x_train-np.mean(x_train,axis=0))
# 	fig,ax=plt.subplots()
# 	plt.hist(all_dists,bins=100,label="dim="+str(x_train.shape[1]))
# 	plt.legend()
# 	plt.savefig('dist_hist_dim='+str(x_train.shape[1]))
# 	d_5, d_50, d_95 = np.percentile(all_dists,5),np.percentile(all_dists,50),np.percentile(all_dists,95)
# 	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
# 	sigma = d_50/((-np.log(0.50))**(1/p))
# 	kernel_train = get_fpkp_kernel_matrix(x_train,x_train,k,p,sigma)
# 	print("Training ...")
# 	clf.fit(kernel_train, y_train)
# 	print("Computing Test Kernel Mat ...")
# 	kernel_test = get_fpkp_kernel_matrix(x_test,x_train,k,p,sigma)
# 	print("Testing ...")
# 	y_pred = clf.predict(kernel_test)
# 	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
# 	return score

def experiment(dim):
	x_train,x_test,y_train,y_test = prep_data(n_samples=3000,n_features=dim)
	rbf_score = test_rbf_kernel(x_train,x_test,y_train,y_test)
	# print("RBF Score = ", rbf_score)
	p_gauss_score = test_p_gaussian_kernel(x_train,x_test,y_train,y_test)
	# print("p-Gauss Score = ", p_gauss_score)
	fp_gauss_score = test_fp_gaussian_kernel(x_train,x_test,y_train,y_test)
	# print("fp-Gauss Score = ", fp_gauss_score)
	# fpkp_gauss_score = test_fpkp_gaussian_kernel(x_train,x_test,y_train,y_test)
	# print("fpkp-Gauss Score = ", fpkp_gauss_score)
	# return (rbf_score,p_gauss_score,fp_gauss_score,fpkp_gauss_score)
	print("Dim=",dim,"Done")
	return (rbf_score,p_gauss_score,fp_gauss_score)

regularization_val = 1000.0

dim_list = [50,100,150,200,250,300,350,400,450,500,550,600]
cores = len(dim_list)
p = Pool(cores)
all_scores = p.map(experiment,dim_list)
p.close()

# print("all_scores=\n")
# print(all_scores)

print("\n\nDone\n")
# print([s[0] for s in all_scores])
# print([s[1] for s in all_scores])
# print([s[2] for s in all_scores])
# print([s[3] for s in all_scores])

fig,ax=plt.subplots()
ax.plot(dim_list,[s[0][0] for s in all_scores],label='RBF Train')
ax.plot(dim_list,[s[1][0] for s in all_scores],label='p-Gauss Train')
ax.plot(dim_list,[s[2][0] for s in all_scores],label='fp-Gauss Train')
# # ax.plot(dim_list,[s[3] for s in all_scores],label='fpkp-Gauss')
ax.plot(dim_list,[s[0][1] for s in all_scores],label='RBF Test')
ax.plot(dim_list,[s[1][1] for s in all_scores],label='p-Gauss Test')
ax.plot(dim_list,[s[2][1] for s in all_scores],label='fp-Gauss Test')
ax.legend()
plt.savefig('test_train_acc_5000pts')
