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

all_dists = np.linalg.norm(x_train-np.mean(x_train,axis=0),axis=1)
d_5,d_50,d_95 = np.percentile(all_dists, 5), np.percentile(all_dists, 50), np.percentile(all_dists, 95)
print(d_5,d_50,d_95)

print("Default RBF")
clf = sklearn.svm.SVR(kernel='rbf',gamma='scale',C=.1)
clf.fit(x_train,y_train)
y_learnt = clf.predict(x_train)
y_pred = clf.predict(x_test)
print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))

print("Gamma Recommended RBF")
clf = sklearn.svm.SVR(kernel='rbf',gamma=-np.log(0.5)/(d_50**2),C=1)
clf.fit(x_train,y_train)
y_learnt = clf.predict(x_train)
y_pred = clf.predict(x_test)
print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))

print("XGB")
clf = XGBRegressor(objective='reg:linear',learning_rate=0.1,max_depth=10,n_estimators=10,reg_lambda=8)
clf.fit(x_train,y_train)
y_learnt = clf.predict(x_train)
y_pred = clf.predict(x_test)
print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))

# print("p-Gaussian")
# def p_gaussian_kernel(distances,p=10.0,sigma=1.0):
# 	kernel_vals = np.exp( - (distances/sigma)**p ) 
# 	return kernel_vals 
# def get_p_kernel_matrix(x1,x2,k,p,sigma):
# 	df1 = lambda x: (np.sum(abs(x)**k))**(1/k)
# 	kernel_mat = np.zeros((len(x1),len(x2)))
# 	for i in (range(len(x1))):
# 		for j in range(len(x2)):
# 			kernel_mat[i,j] = p_gaussian_kernel(df1(x1[i]-x2[j]),p,sigma)
# 	return kernel_mat
# clf = sklearn.svm.SVR(C=0.08,kernel='precomputed')
# k = 2.0
# df1 = lambda x: ((np.sum(abs(x)**k,axis=1))**(1/k) if len(x.shape)>1 else (np.sum(abs(x)**k))**(1/k))
# all_dists = df1(x_train-np.mean(x_train,axis=0))
# d_5, d_50, d_95 = np.percentile(all_dists,5),np.percentile(all_dists,50),np.percentile(all_dists,95)
# p = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
# sigma = d_50/((-np.log(0.50))**(1/p))
# kernel_train = get_p_kernel_matrix(x_train,x_train,k,p,sigma)
# clf.fit(kernel_train, y_train)
# y_learnt = clf.predict(kernel_train)
# kernel_test = get_p_kernel_matrix(x_test,x_train,k,p,sigma)
# y_pred = clf.predict(kernel_test)
# print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
# print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))


print("GP RBF")
import sklearn.gaussian_process as gp
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)
model.fit(x_train, y_train)
params = model.kernel_.get_params()
y_learnt, _ = model.predict(x_train, return_std=True)
y_pred, _ = model.predict(x_test, return_std=True)
print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))


print("GP Matern")
import sklearn.gaussian_process as gp
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.Matern()
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-10, normalize_y=True)
model.fit(x_train, y_train)
params = model.kernel_.get_params()
y_learnt, _ = model.predict(x_train, return_std=True)
y_pred, _ = model.predict(x_test, return_std=True)
print("Pearson Correlation: ","Train:",np.corrcoef(y_train,y_learnt)[0,1],"Test:",np.corrcoef(y_test,y_pred)[0,1])
print("Error:","Train:",mean_squared_error(y_true=y_train,y_pred=y_learnt),"Test:",mean_squared_error(y_true=y_test,y_pred=y_pred))


# plt.hist(all_dists,bins=100)
# plt.show()