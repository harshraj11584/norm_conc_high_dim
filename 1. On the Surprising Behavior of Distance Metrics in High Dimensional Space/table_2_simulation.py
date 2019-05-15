import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

#List of k to try for L-k norms
k_values = np.array([2.0, 1.0])
#Number of Trials
t=10000
#No of points
N=10
#Dimensions
D=np.arange(start=0,stop=100,step=5,dtype=int)

print("\nTrials=",t,"N=",N)

#dimensionwise freq array
freq = np.zeros(len(D))

for dim_ind in range(len(D)) :
	for trial_num in range(t):
		#generating 10 datapoints
		data = np.random.uniform(low=0.0,high=1.0,size=(N,D[dim_ind]))
		#calculating Ud for each datapoint
		Ud = (max(np.linalg.norm(data,ord=2,axis=1))-min(np.linalg.norm(data,ord=2,axis=1)))/(min(np.linalg.norm(data,ord=2,axis=1))+0.0000000001)
		#calculating Td for each datapoint
		Td = (max(np.linalg.norm(data,ord=1,axis=1))-min(np.linalg.norm(data,ord=1,axis=1)))/(min(np.linalg.norm(data,ord=1,axis=1))+0.0000000001)
		freq_trial = np.sum(Ud<Td)
		freq[dim_ind] = freq[dim_ind] + freq_trial

#converting freq to probability
prob = freq/(t*1.0)

plt.plot(D,prob, 'o')
plt.xlabel("Dimensions")
plt.ylabel("Prob(Ud<Td)")
plt.title("N="+str(N)+", trials="+str(t))
plt.grid()
fig.savefig('table_2'+'.png')
plt.show()