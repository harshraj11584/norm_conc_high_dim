import pandas as pd
import numpy as np
import sklearn.datasets


def prep_data():
	X,Y = sklearn.datasets.make_classification(n_samples=50000,n_features=1000,n_informative=950,n_classes=5,n_clusters_per_class=2,random_state=42)

	# d = df = pd.read_csv('/home/harsh/Desktop/Conc Norms/Repo/Presentation 1/Application/news20_class/news20.tar.gz', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
	# print(d)
	print(X.shape,Y.shape)

dataset = prep_data()