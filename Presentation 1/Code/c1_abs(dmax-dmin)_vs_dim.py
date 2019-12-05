import numpy as np
import matplotlib.pyplot as plt

#List of k to try for L-k norms
k_values = np.array([3.0, 2.0, 1.0, 2.0/3.0, 2.0/5.0])

#Number of Datapoints
N=1000

#Max number of dimensions to try
D=200

#data_array stores N datapoints each for dims 1 to D
data = [[0] * N for i in range(D)]
#data[i,j] = jth datapoint of i+1 dim data

print("Generating Data...")
#Generating Data from uniform distribution in (0,1)
for d in range(D):
	for n in range(N):
		data[d][n] = np.random.uniform(low=0.0, high=1.0, size=d+1)
print("Done\n")

norms = np.zeros((len(k_values),D,N))
#norms[x,y,w] = k_values[x] norm of w'th y dimensional datapoint 

y = np.zeros((len(k_values),D))
# y = abs(Dmax - Dmin)

for x in range(len(k_values)):
	print("Finding L",k_values[x],"norm")
	fig, ax = plt.subplots()
	for d in range(D):
		for n in range(N):
			norms[x,d,n] = np.linalg.norm(data[d][n],ord=k_values[x])
		y[x,d] = abs(max(norms[x,d]) - min(norms[x,d]))
	print("Done\n")
	ax.plot(np.arange(start=1,stop=D+1,step=1), y[x,:], label="$L_{"+str(k_values[x])[:4]+"}$ norm")
	ax.set_xlabel("Dimension d")
	ax.set_ylabel("|Dmax-Dmin| for $L_{"+str(k_values[x])[:4]+"}$ norm")
	plt.legend()
	plt.grid()
	fig.savefig("../Graphs/g1_|Dmax-Dmin| vs Dimension for D=$L_{"+str(k_values[x])[:4]+"}$ norm"+'.png')

# plt.show()
