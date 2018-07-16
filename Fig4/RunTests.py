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

for i in range(105):
	print("DataSet " + str(i) + "is being analysed")
	folderName = "DataSet-" + str(i)
	if(os.path.exists(folderName + '/Completed.txt')):
		continue
	if(not os.path.isdir(folderName)):
		os.mkdir(folderName)
	
	meta_file_url = 'http://webdav.tuebingen.mpg.de/cause-effect/pair'+ "{0:0=4d}".format(i) + '_des.txt'
	plot_url = 'http://webdav.tuebingen.mpg.de/cause-effect/pair'+ "{0:0=4d}".format(i) + '.pdf'
	data_url = 'http://webdav.tuebingen.mpg.de/cause-effect/pair'+ "{0:0=4d}".format(i) + '.txt'

	if(download_files):
		if(not os.path.exists('./' + folderName +'/pair'+ "{0:0=4d}".format(i) + '_des.txt')):
			wget.download(meta_file_url, out = folderName + '/pair'+ "{0:0=4d}".format(i) + '_des.txt')

		if(not os.path.exists('./' + folderName + '/pair'+ "{0:0=4d}".format(i) + '.pdf')):
			wget.download(plot_url, out = folderName + '/pair'+ "{0:0=4d}".format(i) + '.pdf')

	if(not os.path.exists('./' + folderName + '/pair'+ "{0:0=4d}".format(i) + '.txt')):
		wget.download(data_url, out = folderName + '/pair'+ "{0:0=4d}".format(i) + '.txt')

	A = []
	B = []
	with open(folderName + '/pair'+ "{0:0=4d}".format(i) + '.txt') as f:
		data = f.readlines()
	if(i == int(sys.argv[1])):
		f = open('./log.txt','w')
	else:
		f = open('./log.txt','a')
	try:
		for row in data:
			row_data = row[:-1]
			row_data = row_data.strip()
			if(row_data == ''):
				continue
			try:
				a, b = row_data.split(' ')
			except ValueError:
				try:
					a, b = row_data.split('\t')
				except ValueError:
					try:
						a, b = row_data.split('  ')
					except:
						try:
							a, b = row_data.split()
						except ValueError:
							raise	

			A.append(float(a))
			B.append(float(b))
		f.write('Dataset ' + str(i) + ": Parsed successfully\n")
	except Exception as e:
		e = sys.exc_info()[0]
		f.write('Dataset ' + str(i) + ":" + str(e) + "\n")
		continue
	if(len(A) > 2300):
		longcount += 1
		print("Long Dataset, Dataset " + str(i) + "   " + str(len(A)))
		continue

	try:
		endIter, correlation_matrix, scoreX, scoreY = ba.analyse_data(A,B, np.arange(0.1,1.0,0.1), folderName)
	except ZeroDivisionError:
		with open("done.txt",'a') as d:
			d.write("Dataset " + str(i) + ": Had ZeroDivisionError hence passing,Thread " + str(sys.argv[3]))
		with open(folderName + "/Completed.txt",'w') as f:
			f.close()
		continue
	pearsonCorr = stats.pearsonr(A,B)

	with open(folderName + '/data.pickle','wb') as d:
		pickle.dump([A,B, correlation_matrix, pearsonCorr, scoreX, scoreY], d)

	if(scoreX > scoreY):
		file =  open(folderName + "/X.txt",'w')
		file.close()
	else:
		file = open(folderName + "/Y.txt",'w')
		file.close()
	if(endIter):
		with open(folderName + "/Completed.txt",'w') as f:
			f.close()
		plt.close()
		sns.set_style("whitegrid")
		ax = sns.heatmap(correlation_matrix, vmin = 0, vmax = 1, center = 0.5)
		ax.set_xlabel("Variance Ratio for Y")
		ax.set_ylabel("Variance Ratio for X")
		plt.savefig("./Heatmaps/heatmap_"+str(i)+".png")
		with open("done.txt",'a') as d:
			d.write("Dataset " + str(i) + ": Completed by Thread " + str(sys.argv[3]))
	else:
			with open(folderName + "/Completed_Coords.txt",'w') as f:
				f.write(str(len(correlation_matrix)) + " " + str(len(correlation_matrix[-1])))
	exit()