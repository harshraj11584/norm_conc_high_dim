import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import sklearn.model_selection
import sklearn.datasets
import pandas as pd 
import sklearn.metrics

# Prep Data 

# data = pd.read_csv('dataset_59_ionosphere.csv')
# print(data.head())
# Y = data['class']
# X = data.drop(['class'],axis=1)
# X = np.array(X)
# Y = np.array(preprocessing.LabelEncoder().fit_transform(Y))
# x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.7,random_state=42)

n_features = 300
X,Y = sklearn.datasets.make_classification(n_samples=1000,n_features=n_features,n_informative=int(0.8*n_features),n_classes=5,n_clusters_per_class=5,random_state=42)
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.5,random_state=42)

# Normal Nearest Neighbours
k_vals = [2,5,10,20,30,50]
scores = []
print("Normal k-NN")
for k in k_vals:
	knn = KNeighborsClassifier(algorithm='brute', 
	                     metric='minkowski',
	                     n_jobs=-1, 
	                     n_neighbors=k, 
	                     p=2,
	                     weights='uniform')
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	scores.append(score)
	# print(sklearn.metrics.classification_report(y_test, y_pred))
print("k=\n",k_vals,"scores=\n",scores)


# RBF Kernel Nearest Neighbours
k_vals = [2,5,10,20,30,50]
scores = []
print("RBF k-NN")

all_dists = []
for i in range(len(x_train)):
	for j in range(i+1,len(x_train)):
		all_dists.append(np.linalg.norm(x_train[i]-x_train[j]))
all_dists = np.array(all_dists)
d_50= np.percentile(all_dists,50)
sigma = (d_50**2)/((-np.log(0.50)))

for k in k_vals:
	knn = KNeighborsClassifier(algorithm='brute', 
	                     metric='minkowski',
	                     n_jobs=-1, 
	                     n_neighbors=k, 
	                     p=2,
	                     weights=lambda d : np.exp(-(d**2)/sigma))
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	scores.append(score)
	# print(sklearn.metrics.classification_report(y_test, y_pred))
print("k=\n",k_vals,"scores=\n",scores)


# p-Gaussian Kernel Nearest Neighbours
k_vals = [2,5,10,20,30,50]
scores = []
print("p-Gaussian k-NN")
all_dists = []
for i in range(len(x_train)):
	for j in range(i,len(x_train)):
		all_dists.append(np.linalg.norm(x_train[i]-x_train[j]))
# all_dists = np.linalg.norm(x_train-np.mean(x_train,axis=0),axis=1)
all_dists = np.array(all_dists)
d_5, d_50, d_95 = np.percentile(all_dists,5),np.percentile(all_dists,50),np.percentile(all_dists,95)
p_sugg = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
sigma = d_50/((-np.log(0.50))**(1/p_sugg))
for k in k_vals:
	knn = KNeighborsClassifier(algorithm='brute', 
	                     metric='minkowski',
	                     n_jobs=-1, 
	                     n_neighbors=k, 
	                     p=2,
	                     weights=lambda d : np.exp(-((d/sigma)**(p_sugg)))
	                     )
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	scores.append(score)
	# print(sklearn.metrics.classification_report(y_test, y_pred))
print("k=\n",k_vals,"scores=\n",scores)


# frac metric RBF Kernel Nearest Neighbours
# k_vals = [2,5,10,20,30,50]
# scores = []
# print("frac metric RBF k-NN")
# f=0.5
# all_dists = []
# for i in range(len(x_train)):
# 	for j in range(i+1,len(x_train)):
# 		all_dists.append(np.linalg.norm(x_train[i]-x_train[j],ord=f))
# all_dists = np.array(all_dists)
# d_50= np.percentile(all_dists,50)
# sigma = (d_50**2)/((-np.log(0.50)))
# for k in k_vals:
# 	knn = KNeighborsClassifier(algorithm='brute', 
# 	                     metric=lambda a,b : np.linalg.norm(a-b,ord=f),
# 	                     n_jobs=-1, 
# 	                     n_neighbors=k, 
# 	                     weights=lambda d : np.exp(-(d**2)/sigma))
# 	knn.fit(x_train, y_train)
# 	y_pred = knn.predict(x_test)
# 	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
# 	scores.append(score)
# 	# print(sklearn.metrics.classification_report(y_test, y_pred))
# print("k=\n",k_vals,"scores=\n",scores)



# frac metric^frac RBF Kernel Nearest Neighbours
k_vals = [2,5,10,20,30,50]
scores = []
print("frac metric^frac RBF k-NN")
f=0.5
all_dists = []
for i in range(len(x_train)):
	for j in range(i+1,len(x_train)):
		all_dists.append(np.linalg.norm(x_train[i]-x_train[j],ord=f)**f)
all_dists = np.array(all_dists)
d_50= np.percentile(all_dists,50)
sigma = (d_50**2)/((-np.log(0.50)))
for k in k_vals:
	knn = KNeighborsClassifier(algorithm='brute', 
	                     metric=lambda a,b : np.linalg.norm(a-b,ord=f)**f,
	                     n_jobs=-1, 
	                     n_neighbors=k, 
	                     weights=lambda d : np.exp(-(d**2)/sigma))
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	scores.append(score)
	# print(sklearn.metrics.classification_report(y_test, y_pred))
print("k=\n",k_vals,"scores=\n",scores)



# p AND frac metric^frac RBF Kernel Nearest Neighbours
k_vals = [2,5,10,20,30,50]
scores = []
print("frac metric^frac AND p-Gaussian k-NN")
f=0.5
all_dists = []
for i in range(len(x_train)):
	for j in range(i+1,len(x_train)):
		all_dists.append(np.linalg.norm(x_train[i]-x_train[j],ord=f)**f)
all_dists = np.array(all_dists)
d_5, d_50, d_95 = np.percentile(all_dists,5),np.percentile(all_dists,50),np.percentile(all_dists,95)
p_sugg = np.log(np.log(0.05)/np.log(0.95)) / np.log(d_95/d_5) 
sigma = d_50/((-np.log(0.50))**(1/p_sugg))
for k in k_vals:
	knn = KNeighborsClassifier(algorithm='brute', 
	                     metric=lambda a,b : np.linalg.norm(a-b,ord=f)**f,
	                     n_jobs=-1, 
	                     n_neighbors=k, 
	                     weights=lambda d : np.exp(-((d/sigma)**p_sugg))
	                     )
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	scores.append(score)
	# print(sklearn.metrics.classification_report(y_test, y_pred))
print("k=\n",k_vals,"scores=\n",scores)


# SIMILARITY BASED Nearest Neighbours
k_vals = [2,5,10,20,30,50]
scores = []
print("SIMILARITY k-NN")

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

def get_distance(x,y,grid,p=2.0):
	dim = grid.shape[0]
	x = x.reshape((dim,))
	y = y.reshape((dim,))
	kd = grid.shape[1]
	c = (np.abs(x-y))
	b = np.zeros((dim,))
	a = np.zeros((dim,))
	for d in range(dim):
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
			# b[d] = 1.0
			a[d] = 1.0
	sim = np.sum((a-b*c)**p)**(1/p)
	# similarity = sim/((np.sum(np.ones(dim)**p))**(1./p))
	# dist = (X.shape[1])**(1/p) - sim
	dist = 1/(sim+1e-5) 
	# if np.isnan(dist):
	# 	print(np.sum(a-b*c))
	# 	print(dist)
	return dist

kd = 1.0*X.shape[1]
for k in k_vals:
	grid = prep_grids(x_train,kd)
	knn = KNeighborsClassifier(algorithm='brute', 
	                     metric=get_distance,
	                     metric_params = { 'grid': grid },
	                     n_jobs=-1, 
	                     n_neighbors=k, 
	                     weights='distance'
	                     )
	knn.fit(x_train, y_train)
	y_pred = knn.predict(x_test)
	score = sklearn.metrics.accuracy_score(y_true=y_test,y_pred=y_pred)
	scores.append(score)
	print("k=",k,"score=",score)
	# print(sklearn.metrics.classification_report(y_test, y_pred))
print("k=\n",k_vals,"scores=\n",scores)
