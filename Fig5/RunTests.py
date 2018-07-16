from urllib.request import urlopen
import os
import wget
import pickle
import Bayesian_Analysis as ba
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

longcount = 0
total = 105
download_files = True
for i in [5]:
	folderName = "DataSet-" + str(i)
	A = []
	B = []
	with open(folderName + '/X.pickle','rb') as f:
		A = list(pickle.load(f))
	with open(folderName + '/Y.pickle','rb') as f:
		B = list(pickle.load(f))
	try:
		endIter, correlation_matrix, scoreX, scoreY = ba.analyse_data(A,B, np.arange(0.1,1.0,0.1), folderName)
	except ZeroDivisionError:
		with open("done.txt",'a') as d:
			d.write("Dataset " + str(i) + ": Had ZeroDivisionError hence passing,Thread " + str(sys.argv[3]))
		with open(folderName + "/Completed.txt",'w') as f:
			f.close()
	pearsonCorr = stats.pearsonr(A,B)

	with open(folderName + '/data.pickle','wb') as d:
		pickle.dump([A,B, correlation_matrix, pearsonCorr, scoreX, scoreY], d)