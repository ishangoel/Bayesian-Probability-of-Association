from urllib.request import urlopen
import os
import wget
import pickle
import Bayesian_Analysis as ba
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys
import pandas as pd
import difflib
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
import math

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    c_X = np.histogram(x,bins)[0]
    c_Y = np.histogram(y,bins)[0]
    entX = stats.entropy(c_X)
    entY = stats.entropy(c_Y)
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi/(math.sqrt(entX*entY))

ratio_X = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
ratio_Y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
ratioReverse = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
def findeffect(i):
	with open(folderName + "/pair"+"{0:0=4d}".format(i)+"_des.txt",'r') as f:
		content = f.readlines()
		content = [x.strip() for x in content]
		flag = False
		for j in range(len(content)):
			line = content[j]
			flag = True
			if(difflib.SequenceMatcher(None, line, "Ground Truth:").ratio() > 0.8):

				break
		if(flag):
			result = content[j+1]
			k = 2
			while(result == ''):
				result = content[j+k]
				k += 1
			result = result[-1]
			if(result.lower() == 'x')or(result.lower() == 'y'):
				return result
			else:
				return False
		else:
			return False


correctCausal = 0
incorrectCausal = 0
insufficientCausal = 0
totalCausal = 0
longcount = 0
total = 105 
download_files = True
logFile = './Overalllog.txt'
heatMapsFolder = './Heatmaps'
precisions = np.arange(0.1,1.0,0.1)

plt.close()
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'gainsboro'
fig = plt.figure(figsize = [7.75,11.62], dpi = 500) 
plt.subplots_adjust(top=0.97,bottom=0.05,right=0.9,left=0.07 ,hspace=0.3,wspace=0.3)
# outerGS = gridspec.GridSpec(10,10, wspace = 0.1, hspace = 0.1, width_ratios = [0.01,1,1,1,1,1,1,1,1,0.01], height_ratios = [0.01,1,1,1,1,1,1,1,1,0.01])
# outerGS = gridspec.GridSpec(15,5, wspace =0.2, hspace = 0.2)
outerGS = gridspec.GridSpec(15,10, wspace = 0.15, hspace = 0.17)
cbar_ax = fig.add_axes([.925, .35, .022, .3])
x = 0
y = 0
if(not os.path.isdir(heatMapsFolder)):
	os.mkdir(heatMapsFolder)
for i in range(1,107):
	print(i)
	folderName = "./DataSet-" + str(i)
	hasFolder = os.path.isdir(folderName)
	hasLogFile = os.path.isfile(logFile)
	if(not hasFolder):
		print('Does not have folder')
		if(hasLogFile):
			with open(logFile, 'a') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Dataset folder was not found")
		else:
			with open(logFile, 'w') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Dataset folder was not found")
		continue

	hasDataFile = os.path.isfile(folderName + "/data.pickle")
	hasOnlineData = os.path.isfile(folderName + "/pair"+"{0:0=4d}".format(i)+"_des.txt")

	if(not hasOnlineData):
		if(hasLogFile):
			with open(logFile, 'a') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Folder was made. But data was not downloaded")
		else:
			with open(logFile, 'w') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Folder was made. But data was not downloaded")
		continue

	if(not hasDataFile):
		if(hasLogFile):
			with open(logFile, 'a') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Data was not processed.")
		else:
			with open(logFile, 'w') as f: 
				f.write("\n")
				f.write("Dataset " + str(i) + ":Data was not processed.")
		continue

	with open(folderName + "/data.pickle", 'rb') as f:
		A,B, correlation_matrix, pearsonCorr, scoreX, scoreY = pickle.load(f)

	MI = calc_MI(A,B,100)

	if((len(correlation_matrix) < 9)or(len(correlation_matrix[-1]) < 9)):
		if(hasLogFile):
			with open(logFile, 'a') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Data was incomplete")
		else:
			with open(logFile, 'w') as f:
				f.write("\n")
				f.write("Dataset " + str(i) + ":Data was incomplete")
		continue

	effect = findeffect(i)
	correctFlag = 0
	scoreX = 0
	scoreY = 0
	for c1 in range(len(correlation_matrix)):
		for c2 in range(len(correlation_matrix)):
			scoreX += precisions[c1]*correlation_matrix[c1][c2]
			scoreY += precisions[c2]*correlation_matrix[c1][c2]
	if(abs(scoreX - scoreY) > 0.01):
		if(scoreX > scoreY):
			predictedResult = 'x'
		else:
			predictedResult = 'y'
	else:
		predictedResult = False

	totalCausal += 1
	if(predictedResult):
		if(effect == predictedResult):
			correctCausal += 1
			correctFlag = True
		else:
			incorrectCausal += 1
	else:
		correctFlag = 2
		insufficientCausal += 1

	hMap = plt.subplot(outerGS[x,y])
	sPlot = plt.subplot(outerGS[x,y+1])

	hMap = sns.heatmap(correlation_matrix, vmin = 0, vmax = 1, ax = hMap, xticklabels = ratio_X, yticklabels = ratioReverse, cbar = (x + y) == 0, cbar_ax = None if (x + y) else cbar_ax,square = True)
	colors = ['red','dodgerblue','forestgreen']
	sPlot.scatter(A,B,s = 0.03, color = 'black',alpha = 0.8)
	labels = [item.get_text() for item in hMap.get_xticklabels()]
	settochange = [1,7]
	changed = {1:"0.2",4:"0.5",7:"0.8"}
	for lc in range(len(labels)):
		if(lc in settochange):
			labels[lc] = changed[lc]
		else:
			labels[lc] = ""

	hMap.set_xticklabels(labels)
	labels = [item.get_text() for item in hMap.get_yticklabels()]
	settochange = [1,7]
	changed = {1:"0.8",4:"0.5",7:"0.2"}
	for lc in range(len(labels)):
		if(lc in settochange):
			labels[lc] = changed[lc]
		else:
			labels[lc] = ""

	texts = ['I','C','U']
	hMap.set_yticklabels(labels)
	sPlot.set_title(" PC:" + '%.2f'%pearsonCorr[0], fontsize = 7, y = 0.87, x = 0.7)
	hMap.set_title(texts[correctFlag] + "       MI:" + '%.2f'%MI, fontsize = 7, y = 0.9, x = 0.5)
	hMap.tick_params(labelleft='off',labelbottom='off',labelsize = 7)
	sPlot.tick_params(labelleft='off',labelbottom='off',labelsize = 7)

	if((x == 14)):
		hMap.tick_params(labelbottom='on')
		hMap.set_xlabel(r"$\alpha_Y^{'}$",fontsize = 8, fontweight = 'bold')
		sPlot.set_xlabel("Y",fontsize = 8, fontweight = 'bold')
	if((y ==0)):
		hMap.tick_params(labelleft='on')
		hMap.set_ylabel(r"$\alpha_X^{'}$",fontsize = 8, fontweight = 'bold')
	if(y == 8):
		y = 0
		x += 1
	else:
		y += 2

plt.axis('off')
cbar_ax.tick_params(labelsize = 7)
plt.savefig('Figure4.png',dpi = 500)


print(float(correctCausal)/float(totalCausal))
print(float(incorrectCausal)/float(totalCausal))
print(float(insufficientCausal)/float(totalCausal))

print(float(correctCausal))
print(float(incorrectCausal))
print(float(insufficientCausal))
