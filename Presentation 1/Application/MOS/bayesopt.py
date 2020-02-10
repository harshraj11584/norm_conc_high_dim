from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import sklearn.datasets
import sklearn.model_selection
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool 
from xgboost import XGBRegressor

with h5py.File('MOS_Scores.mat', 'r') as f:
    for k,v in f.items():
    	Y = (np.array(v))
    	Y = Y.reshape(Y.shape[1])
with h5py.File('konvid_features.mat', 'r') as f:
    for k,v in f.items():
    	X = (np.array(v)).T
    	X = StandardScaler().fit_transform(X)

print(X.shape,Y.shape)
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.2,shuffle=True,random_state=42)

def objective_function(k,c):
	# print("p-Gaussian")
	def get_p_kernel_matrix(x1,x2,k,p,sigma):
		df1 = lambda x: (np.sum(abs(x)**k))**(1/k)
		kernel_mat = np.zeros((len(x1),len(x2)))
		for i in (range(len(x1))):
			kernel_mat[i,:] = np.array([np.exp(-(np.linalg.norm(x1[i]-x2[j],ord=k)/sigma)**p) for j in range(len(x2))])
		return kernel_mat
	clf = sklearn.svm.SVR(C=10**c,kernel='precomputed')
	df1 = lambda x: ((np.sum(abs(x)**k,axis=1))**(1/k) if len(x.shape)>1 else (np.sum(abs(x)**k))**(1/k))
	all_dists = df1(x_train-np.mean(x_train,axis=0))
	d_5, d_50, d_95 = np.percentile(all_dists,5),np.percentile(all_dists,50),np.percentile(all_dists,95)
	p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
	sigma = d_50/((-np.log(0.50))**(1/p))
	kernel_train = get_p_kernel_matrix(x_train,x_train,k,p,sigma)
	clf.fit(kernel_train, y_train)
	# y_learnt = clf.predict(kernel_train)
	kernel_test = get_p_kernel_matrix(x_test,x_train,k,p,sigma)
	y_pred = clf.predict(kernel_test)
	# print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
	# print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))
	return np.corrcoef(y_test,y_pred)[0,1] 


#Boundary Region
pbounds = {'k': (0.1, 2.1), 'c': (-3, 3)}

optimizer = BayesianOptimization(
	f=objective_function,
	pbounds=pbounds,
	random_state=42
	)

optimizer.maximize(
	init_points=5,
	n_iter=3
	)

print(optimizer.max)