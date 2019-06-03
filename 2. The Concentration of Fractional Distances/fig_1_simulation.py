import numpy as np
import matplotlib.pyplot as plt

#Number of Datapoints
N=100000

#Max number of dimensions to try
D=100

#data_array stores N datapoints each for dims 1 to D
data = [[0] * N for i in range(D)]
#data[i,j] = jth datapoint of i+1 dim data

print("Generating Data...")
#Generating Data from uniform distribution in (0,1)
for d in range(D):
	for n in range(N):
		data[d][n] = np.random.uniform(low=0.0, high=1.0, size=d+1)
print("Done\n")

norms = np.zeros((D,N))
#norms[y,w] = Euclidean norm of w'th y dimensional datapoint

min_obs = np.zeros(D)
max_obs = np.zeros(D)
avg_obs = np.zeros(D)
std_dev = np.zeros(D) 
max_poss = np.zeros(D)

print("Finding Euclidean norms")
fig, ax = plt.subplots()
for d in range(D):
	for n in range(N):
		norms[d,n] = np.linalg.norm(data[d][n],ord=2)
	min_obs[d] = np.amin(norms[d])
	max_obs[d] = np.amax(norms[d])
	std_dev[d] = np.std(norms[d])
	avg_obs[d] = np.mean(norms[d])
	max_poss[d] = (d+1)**0.5
print("Done\n")

plt.plot(np.arange(1,D+1,1),min_obs,label="min_obs")
plt.plot(np.arange(1,D+1,1),max_obs,label="max_obs")
plt.plot(np.arange(1,D+1,1),avg_obs,label="avg_obs")
plt.plot(np.arange(1,D+1,1),avg_obs+std_dev,label="avg+std")
plt.plot(np.arange(1,D+1,1),avg_obs-std_dev,label="avg-std")
plt.plot(np.arange(1,D+1,1),max_poss,label="max_poss")


plt.legend()
plt.grid()
fig.savefig('fig_1_(a)(b).png')
plt.show()