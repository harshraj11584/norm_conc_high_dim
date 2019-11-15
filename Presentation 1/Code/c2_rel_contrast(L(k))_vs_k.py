import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

#List of k to try for L-k norms
k_values = np.concatenate((np.arange(start=0.1,stop=1,step=0.25),np.arange(start=1.0,stop=11.0,step=1)))
markers=['^', 'o', '*']
#No of points
N=np.array([100,1000,10000])
#Dimensions
dim=20 

for n in N:
	#generating 10 datapoints
	data = np.random.uniform(low=0.0,high=1.0,size=(n,dim))
	rel_contrast = np.array([(max(np.linalg.norm(data,ord=k,axis=1))-min(np.linalg.norm(data,ord=k,axis=1)))/(min(np.linalg.norm(data,ord=k,axis=1))+0.0000000001) for k in k_values])
	plt.plot(k_values,rel_contrast,linestyle='--', marker=markers[int(np.log10(n)-2)],label="N="+str(n)+" datapoints")

plt.xlabel("k for D=L(k) norm") 
plt.ylabel("Relative Contrast RC=|Dmax-Dmin|/Dmin") 
plt.legend() 
plt.grid() 
fig.savefig('../Graphs/g2_rel_contrast(L(k))_vs_k'+'.png') 
# plt.show() 