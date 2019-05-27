import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
fig, ax = plt.subplots()

#Generating Synthetic Data 
dim = 2
n_clusters = 6
N = 10
means=np.random.uniform(low=0.0,high=100.0,size=n_clusters)
means.sort()
data=[np.random.normal(loc=m,scale=1,size=(N,dim)) for m in means]
data = np.array(data)
data = data.reshape((N*n_clusters,dim))
#Actual Labels
y_act = np.array([[i+1]*N for i in range(len(means))]).reshape((N*n_clusters))

print("data\n",data.shape)
print(data)
print()
print("y_act\n",y_act.shape)
print(y_act)


