import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import BayesianQuantitation as bq
import scipy.stats as stats
import pickle
import matplotlib.gridspec as gridspec
import os.path
import os
import math
from random import shuffle
def makeDir(name):
	if(not os.path.isdir(name)):
		os.mkdir(name)
	return name

print("Starting...")
ratio_X = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
ratio_Y = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
est_ratio_X = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
est_ratio_Y = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
peelSize = 100
trials = 1
variance = 10.0
graphsDir = makeDir("./UnderstandingTheParameter_Graphs")
pickleDir  = makeDir("./UnderstandingTheParameter_PickleFiles")
finalList = []

for ratX in ratio_X:
	for ratY in ratio_Y:
		finalList.append([ratX,ratY])

shuffle(finalList)
for ratXY in finalList:
	ratX = ratXY[0]
	ratY = ratXY[1]
	pickleFile = pickleDir + "/data_" + str(ratX) + "_" + str(ratY) + ".pickle"
	graphFile = graphsDir + "/HeatMap_" + str(ratX) + "_" + str(ratY) + ".png"
	if(os.path.exists(pickleFile)):
		continue	
	print("\n\n\n\n\n")
	print(ratX, ratY)
	param_X = [0,math.sqrt(variance*variance*ratX)]
	param_Y = [0,math.sqrt(variance*variance*ratY)]
	param_e1 = [0,math.sqrt(variance*variance*(1 - ratX))]
	param_e2 = [0,math.sqrt(variance*variance*(1 - ratY))]
	peel1 = np.random.normal(size = peelSize, loc = param_X[0], scale = param_X[1])
	peel2 = np.random.normal(size = peelSize, loc = param_Y[0], scale = param_Y[1])
	error1 = np.random.normal(size = peelSize, loc = param_e1[0], scale = param_e1[1])
	error2 = np.random.normal(size = peelSize, loc = param_e2[0], scale = param_e2[1])

	assoc_list1 = np.add(peel1, error1)
	assoc_list2 = np.add(peel1, error2)

	non_assoc_list1 = np.add(peel1,error1)
	non_assoc_list2 = np.add(peel2,error2)

	assocMatrix = []
	nonassocMatrix = []
	assocCorrelation = np.abs(stats.pearsonr(assoc_list1, assoc_list2)[0])
	nonassocCorrelation = np.abs(stats.pearsonr(non_assoc_list1, non_assoc_list2)[0])
	index = -1
	for estRatX in est_ratio_X:
		assocMatrix.append([])
		nonassocMatrix.append([])
		index += 1
		for estRatY in est_ratio_Y:
			print("\n\n")
			print(estRatX, estRatY)

			estParam_X = [0,math.sqrt(variance*variance*estRatX)]
			estParam_Y = [0,math.sqrt(variance*variance*estRatY)]
			estParam_e1 = [0,math.sqrt(variance*variance*(1 - estRatX))]
			estParam_e2 = [0,math.sqrt(variance*variance*(1 - estRatY))]
			assocBQ = bq.calculateProbability(assoc_list1, assoc_list2, estParam_X, estParam_Y, estParam_e1, estParam_e2)
			print("\n\n")
			nonassocBQ = bq.calculateProbability(non_assoc_list1, non_assoc_list2, estParam_X, estParam_Y, estParam_e1, estParam_e2)
			assocMatrix[index].append(assocBQ)
			nonassocMatrix[index].append(nonassocBQ)
	print("Can't close")
	with open(pickleFile,'wb') as f:
		pickle.dump([assocMatrix, nonassocMatrix, assocCorrelation, nonassocCorrelation],f)

	plt.close()
	plt.figure(figsize = [8,4])
	gs = gridspec.GridSpec(1,2)
	plt.subplot(gs[0,0])
	plt.title("Associated",fontsize = 8)
	sns.heatmap(assocMatrix, vmin = 0, vmax = 1, xticklabels = est_ratio_X, yticklabels = est_ratio_Y, center = 0.5)
	plt.subplot(gs[0,1])
	plt.title("Non-associated",fontsize = 8)
	sns.heatmap(nonassocMatrix, vmin = 0, vmax = 1, xticklabels = est_ratio_X, yticklabels = est_ratio_Y, center = 0.5)
	plt.tight_layout()
	plt.savefig(graphFile)
	print("Okay to close")