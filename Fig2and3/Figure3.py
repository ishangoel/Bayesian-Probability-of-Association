import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import matplotlib.gridspec as gridspec
import os.path
import os
import math
from random import shuffle
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


print("Starting...")
ratio_X = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
ratio_Y = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
ratioReverse = [0.99,0.9,0.75,0.6,0.45,0.3,0.15,0.01]
peelSize = 100
trials = 1
variance = 10.0
plt.close()
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = 'gainsboro'
fig = plt.figure(figsize = [7.5,7.5], dpi = 600) 
plt.subplots_adjust(top=0.95,bottom=0.1,right=0.9,left=0.1,hspace=0.1,wspace=0.1)
# outerGS = gridspec.GridSpec(10,10, wspace = 0.1, hspace = 0.1, width_ratios = [0.01,1,1,1,1,1,1,1,1,0.01], height_ratios = [0.01,1,1,1,1,1,1,1,1,0.01])
outerGS = gridspec.GridSpec(8,8, wspace = 0.15, hspace = 0.17, width_ratios = [1,1,1,1,1,1,1,1], height_ratios = [1,1,1,1,1,1,1,1])
cbar_ax = fig.add_axes([.915, .35, .022, .3])
# assocInner = gridspec.GridSpecFromSubplotSpec(8,8, subplot_spec = outerGS[0], wspace = 0.1,hspace = 0.1)
# nonAssocInner = gridspec.GridSpecFromSubplotSpec(8,8, subplot_spec = outerGS[1], wspace = 0.1,hspace = 0.1)
headFolder = './UnderstandingTheParameter_PickleFiles'
countX = -1

for ratX in ratio_X:
	countX += 1
	countY = -1
	for ratY in ratio_Y:
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

		MI = calc_MI(non_assoc_list1, non_assoc_list2, 10)

		countY += 1
		print(countX,countY)
		filename = headFolder + "/data_" + str(ratX) + "_" + str(ratY) + ".pickle"
		with open(filename,'rb') as f:
			assocMatrix, nonAssocMatrix, assocCorrelation, nonAssocCorrelation = pickle.load(f)
		assocAx = plt.subplot(outerGS[countX, countY])
		assocAx = sns.heatmap(nonAssocMatrix, vmin = 0, vmax = 1, ax = assocAx, xticklabels = ratio_X, yticklabels = ratioReverse, cbar = (countX + countY) == 0, cbar_ax = None if (countX + countY) else cbar_ax,square = True)
 		# assocAx = sns.heatmap(assocMatrix, vmin = 0, vmax = 1, ax = assocAx, xticklabels = ratio_X, yticklabels = ratio_Y, cbar = False, cbar_ax = None,square = True)
		labels = [item.get_text() for item in assocAx.get_xticklabels()]
		settochange = [1,6]
		changed = {1:"0.15",3:"0.45",6:"0.9"}
		for lc in range(len(labels)):
			if(lc in settochange):
				labels[lc] = changed[lc]
			else:
				labels[lc] = ""
		assocAx.set_xticklabels(labels)

		labels = [item.get_text() for item in assocAx.get_yticklabels()]
		settochange = [1,6]
		changed = {1:"0.9",3:"0.45",6:"0.15"}
		for lc in range(len(labels)):
			if(lc in settochange):
				labels[lc] = changed[lc]
			else:
				labels[lc] = ""
		assocAx.set_yticklabels(labels)
		assocAx.set_title("MI:" + '%.2f'%MI + "  PC:" + '%.2f'%nonAssocCorrelation, fontsize = 7, y = 0.9, x = 0.5)
		assocAx.tick_params(labelleft='off',labelbottom='off',labelsize = 7)

		if((countX == 7)):
			assocAx.tick_params(labelbottom='on')
			assocAx.set_xlabel(r"$\alpha_Y^{'}$",fontsize = 8, fontweight = 'bold')
		if((countY == 0)):
			assocAx.tick_params(labelleft='on')
			assocAx.set_ylabel(r"$\alpha_X^{'}$",fontsize = 8, fontweight = 'bold')

		cbar_ax.tick_params(labelsize = 7)
		# nonAssocAx = plt.subplot(outerGS[countX, countY + 9])
		# sns.heatmap(nonAssocMatrix, vmin = 0, vmax = 1, ax = nonAssocAx, cbar = False)

#plt.tight_layout()
nums = [0.01,0.15,0.3,0.45,0.6,0.75,0.9,0.99]
nums2 = [0.99,0.9,0.75,0.6,0.45,0.3,0.15,0.01]
for i in range(8):
	plt.figtext(0.03,0.107*i + 0.15, nums2[i], fontsize = 7, rotation = 90)
for i in range(8):
	plt.figtext(0.101*i + 0.1372 , 0.03 , nums[i], fontsize = 7)

plt.figtext(0.015 , 0.53 , r"$\alpha_X$", fontsize = 7, rotation = 90)
plt.figtext(0.48, 0.015 , r"$\alpha_Y$", fontsize = 7)

plt.savefig('Figure3.png',dpi = 600)
 