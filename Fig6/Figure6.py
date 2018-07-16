import numpy as np
import scipy.stats as stats
import BayesianQuantitation as bqvd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import math
import numpy
import seaborn as sns

def plotBarGraph(data,filename, colors, labels, xticks, xlabel, ylabel, legend = True):
	plt.close()

	f, ax1 = plt.subplots(1, figsize=(5,5))
	bar_width = 0.5
	bar_l = [i+1 for i in range(len(data[0]))]
	tick_pos = [i for i in bar_l]
	print(len(data))
	for i in range(len(data)):
		if(i==0):
			ax1.bar(bar_l, data[i],width=bar_width,label=labels[i],alpha=0.5,color=colors[i])
		elif(i==1):
			ax1.bar(bar_l, data[i],width=bar_width,bottom=data[i-1],label=labels[i],alpha=0.5,color=colors[i])
		elif(i==2):
			ax1.bar(bar_l, data[i],width=bar_width,bottom=[i+j for i,j in zip(data[i-1],data[i-2])],label=labels[i],alpha=0.5,color=colors[i])
		else:
			ax1.bar(bar_l, data[i],width=bar_width,bottom=[i+j+k for i,j,k in zip(data[i-1],data[i-2], data[i-3])],label=labels[i],alpha=0.5,color=colors[i])


	if(legend):
		plt.legend(frameon=True)
	plt.tight_layout()
	plt.savefig(filename)


functions = [r'$x$',r'$e^{x}$',r'$x^{2}$',r'$\frac{1}{x}$',r'$sin(x)$']
with open('results_nonLinearity_2.pickle','rb') as f:
	listsAssoc,probsAssoc, pearsonCorrelation, MI = pickle.load(f)

plt.close()
plt.style.use('ggplot')
plt.rcParams['axes.facecolor']='white'	
plt.rcParams['grid.color']='gainsboro'

fig = plt.figure(figsize = (6,3), dpi = 300)
gs = gridspec.GridSpec(1,3, width_ratios = [1,0.05,1])
cbar_ax = fig.add_axes([.46, .28, .022, .5])

hMap = plt.subplot(gs[0,0])

sns.heatmap(probsAssoc, vmin = 0, vmax = 1, ax = hMap, cbar = True, square = True, xticklabels = functions, yticklabels = functions, cbar_ax = cbar_ax)
hMap.tick_params(labelsize = 8)
hMap.set_xlabel('Estimated Relationship Function',fontsize = 7)
hMap.set_ylabel('Actual Relationship Function',fontsize = 7)

cbar_ax.tick_params(labelsize = 7)

ax1 = plt.subplot(gs[0,2])
bar_width = 0.2
bar_Pearson = [1.1,2.1,3.1,4.1,5.1]
bar_MI = [1.1 + 0.2,2.1 + 0.2,3.1 + 0.2,4.1 + 0.2,5.1 + 0.2]
bar_Bayesian = [1.1 + 0.2 + 0.2,2.1 + 0.2 + 0.2,3.1 + 0.2 + 0.2,4.1 + 0.2 + 0.2,5.1 + 0.2 + 0.2]
data_ph1 = [probsAssoc[i][i] for i in range(5)]
tick_pos = [1.25,2.25,3.25,4.25,5.25]
ax1.bar(bar_Pearson, pearsonCorrelation,width=bar_width,label='Pearson Correlation Coefficient',alpha=0.5,color='silver')
ax1.bar(bar_MI, MI,width=bar_width,label=r'Mutual Information Content',alpha=0.5,color='dimgray')
ax1.bar(bar_Bayesian, data_ph1,width=bar_width,label=r'$P(H_1)$',alpha=0.5,color='black')
plt.xticks([1.3,2.3,3.3,4.3,5.3], functions, ha = 'center')
ax1.set_yticks([0,0.5,1])
ax1.set_xlabel('Choice of function', fontsize = 7)
ax1.set_ylabel('Results', fontsize = 7)
ax1.set_xlim([min(tick_pos)-2.5*bar_width, max(tick_pos)+2*bar_width])
ax1.legend(frameon = True, fontsize = 7, loc = 'upper right')
ax1.set_ylim([0,1.3])
ax1.tick_params(labelsize = 7)
fig.text(0.03,0.93,'a',fontsize = 7, fontweight = 'bold')
fig.text(0.55,0.93,'b',fontsize = 7, fontweight = 'bold')
plt.tight_layout()
plt.savefig('Figure6.png', dpi = 300)




