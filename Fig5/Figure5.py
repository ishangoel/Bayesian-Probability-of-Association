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
fig = plt.figure(figsize = [7,5], dpi = 300) 
plt.subplots_adjust(top=0.97,bottom=0.08,right=0.83,left=0.15 ,hspace=0.3,wspace=0.3)
# outerGS = gridspec.GridSpec(10,10, wspace = 0.1, hspace = 0.1, width_ratios = [0.01,1,1,1,1,1,1,1,1,0.01], height_ratios = [0.01,1,1,1,1,1,1,1,1,0.01])
# outerGS = gridspec.GridSpec(15,5, wspace =0.2, hspace = 0.2)
outerGS = gridspec.GridSpec(5,6, wspace = 0.15, hspace = 0.17)
cbar_ax = fig.add_axes([.85, .35, .022, .3])
x = 0
y = 0
if(not os.path.isdir(heatMapsFolder)):
	os.mkdir(heatMapsFolder)
for i in [1,5,6,7,8,11,12,13,14,15,16,17,18,19,20]:
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

	with open(folderName + "/data.pickle", 'rb') as f:
		A,B, correlation_matrix, pearsonCorr, scoreX, scoreY = pickle.load(f)

	with open(folderName + "/X.pickle",'rb') as f:
		A_scat = pickle.load(f)

	with open(folderName + "/Y.pickle",'rb') as f:
		B_scat = pickle.load(f)

	MI = calc_MI(A,B,30 )
	hMap = plt.subplot(outerGS[x,y])
	sPlot = plt.subplot(outerGS[x,y+1])

	hMap = sns.heatmap(correlation_matrix, vmin = 0, vmax = 1, ax = hMap, xticklabels = ratio_X, yticklabels = ratio_Y, cbar = (x + y) == 0, cbar_ax = None if (x + y) else cbar_ax,square = True)
	sPlot.scatter(A_scat,B_scat,s = 0.03, color = 'black',alpha = 0.8)
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

	hMap.set_yticklabels(labels)

	hMap.set_yticklabels(labels)
	sPlot.set_title("PC: " + '%.2f'%pearsonCorr[0], fontsize = 7, y = 0.92, x = 0.8)
	hMap.set_title("MI: " + '%.2f'%MI, fontsize = 7, y = 0.92, x = 0.8)
	hMap.tick_params(labelleft='off',labelbottom='off',labelsize = 7)
	sPlot.tick_params(labelleft='off',labelbottom='off',labelsize = 7)

	if((x == 4)):
		hMap.tick_params(labelbottom='on')
		sPlot.tick_params(labelbottom='on')
		hMap.set_xlabel(r"$\alpha_Y^{'}$",fontsize = 8, fontweight = 'bold')
	if((y ==0)):
		hMap.tick_params(labelleft='on')
		hMap.set_ylabel(r"$\alpha_X^{'}$",fontsize = 8, fontweight = 'bold')
	x += 1
	if(x == 5):
		y += 2
		x = 0
cbar_ax.tick_params(labelsize = 7)

plt.savefig('Figure5_new.png', dpi = 300)
