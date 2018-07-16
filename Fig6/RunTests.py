import numpy as np
import scipy.stats as stats
import BayesianQuantitation as bqvd
import pickle
import math
import numpy
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
import math


def normal(x, params):
	return stats.norm.pdf(x, loc = params[0], scale = params[1])

def exponential(x, params):
	return stats.expon.pdf(x, scale = params[0])

def uniform(x, params):
	return stats.uniform.pdf(x,loc = params[0], scale = params[1] - params[0])

def chisquare(x, params):
	return stats.chi2.pdf(x, df = params[0])

def gamma(x, params):
	return stats.gamma.pdf(x, a = params[0], scale = params[1])

def log(x):
	return math.log(x)

def exp(x):
	return math.exp(x)

def xsquare(x):
	return x*x

def xroot(x):
	return math.sqrt(x)

def inverse(x):
	return float(1)/x

def sin(x):
	return math.sin(x)

def sinInv(x):
	return math.asin(x)

def identity(x):
	return x

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    c_X = np.histogram(x,bins)[0]
    c_Y = np.histogram(y,bins)[0]
    entX = stats.entropy(c_X)
    entY = stats.entropy(c_Y)
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi/(math.sqrt(entX*entY))

functions = ['identity','exp','xsquare','inverse','sin']
inv_functions = ['identity','log','xroot','inverse','sinInv']
size = 100
listsAssoc = []
probsAssoc = []
pearsonCorrelation = []
MI = []
for i in range(len(functions)):
	func = functions[i]
	A = np.random.normal(loc = 0, scale = 1, size = 100)
	A2 = []
	for a in A:
		A2.append(eval(func + '(a)'))
	B = np.random.normal(loc = 0, scale = 0.5, size = 100)
	C = np.random.normal(loc = 0, scale = 0.5, size = 100)

	AB = A + B
	A2C = np.array(A2) + np.array(C)
	print(stats.pearsonr(AB, A2C)[0])
	pearsonCorrelation.append(stats.pearsonr(AB, A2C)[0])
	MI.append(calc_MI(AB, A2C,100))
	listsAssoc.append([])
	probsAssoc.append([])
	for j in range(len(functions)):
		print(functions[i], functions[j])
		f_X = normal
		f_Y = normal
		f_e1 = normal
		f_e2 = normal
		f1 = eval(functions[j])
		f2 = eval(inv_functions[j])

		prob_h1, list_probh1 = bqvd.calculateProbability(AB, A2C,[0,1],[np.mean(A2C),np.std(A2C)],[0,1],[0,1], f1, f2)
		listsAssoc[i].append(list_probh1)
		probsAssoc[i].append(prob_h1)

with open('results_nonLinearity_2.pickle','wb') as f:
	pickle.dump([listsAssoc,probsAssoc, pearsonCorrelation, MI],f)