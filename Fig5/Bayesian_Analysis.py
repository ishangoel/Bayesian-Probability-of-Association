import BayesianQuantitation as bq
import numpy as np
import scipy.stats as stats
import math
import pickle
import os

def analyse_data(dataset1,dataset2, precision, folderName = 'RandomFolderName'):

	if(os.path.exists(folderName + '/matrix.pickle')):
		try:
			with open(folderName + '/matrix.pickle','rb') as f:
				A,B,variance_A,variance_B, slope, intercept, slope2, intercept2, start1,start2, matrix, scoreX, scoreY = pickle.load(f)
		except:
			with open(folderName + '/matrix.pickle','rb') as f:
				A,B,variance_A,variance_B, slope, intercept, slope2, intercept2, start1, matrix, scoreX, scoreY = pickle.load(f)
			start2 = 0

	else:
		A = np.array(dataset1) - np.mean(dataset1)
		B = np.array(dataset2) - np.mean(dataset2)

		variance_A = np.var(A)
		variance_B = np.var(B)

		slope, intercept, r, p, e = stats.linregress(A,B)
		slope2, intercept2, r2, p2, e2 = stats.linregress(B,A)

		print(slope,intercept)
		print(slope2,intercept2)
		matrix = []
		start1 = 0
		start2 = 0
		scoreX = 0
		scoreY = 0
	for i in range(start1, len(precision)):
		matrix.append([])
		prec_1 = precision[i]
		for j in range(len(precision)):
			prec_2 = precision[j]
			if((i==start1)and(j < start2)):
				continue
			print (folderName + " " + str(i) + " " + str(j))
			print("\n\n\n")
			parameter = bq.calculateProbability(A, B, [0, math.sqrt(prec_1*variance_A)], [0, math.sqrt((prec_2)*variance_B)], [0, math.sqrt((1-prec_1)*variance_A)], [0, math.sqrt((1-prec_2)*variance_B)], lambda x: intercept + slope*x, lambda x: intercept2 + slope2*x)
			matrix[i].append(parameter)
			scoreX += prec_1*parameter
			scoreY += prec_2*parameter
			if(folderName != 'RandomFolderName'):
				try:
					os.remove(folderName + '/matrix.pickle')
				except OSError:
					pass

				print("Don't Close")
				with open(folderName + '/matrix.pickle','wb') as f:
					if(j == len(precision) - 1):
						pickle.dump([A,B,variance_A,variance_B, slope, intercept, slope2, intercept2, i + 1,0, matrix, scoreX, scoreY], f)
					else:
						pickle.dump([A,B,variance_A,variance_B, slope, intercept, slope2, intercept2, i,j+1, matrix, scoreX, scoreY], f)
				print("Can Close")
	return True, matrix, scoreX, scoreY
