import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

#List of k to try for L-k norms
k_values = np.arange(start=0,stop=10,step=0.5)
#No of points
N=np.array([100,1000,10000])
#Dimensions
dim=20

for n in N:
	#generating 10 datapoints
	data = np.random.uniform(low=0.0,high=1.0,size=(n,dim))
	rel_contrast = np.array([(max(np.linalg.norm(data,ord=k,axis=1))-min(np.linalg.norm(data,ord=k,axis=1)))/(min(np.linalg.norm(data,ord=k,axis=1))+0.0000000001) for k in k_values])
	plt.plot(k_values,rel_contrast,label="N="+str(n))

plt.xlabel("Parameter of Distance Norm")
plt.ylabel("Relative Contrast")
plt.title("Relative Contrasts for uniform distribution (D="+str(dim)+" was used)")
plt.legend()
plt.grid()
fig.savefig('fig_2'+'.png')
plt.show()