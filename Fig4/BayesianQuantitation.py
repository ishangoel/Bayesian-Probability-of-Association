from scipy.integrate import quad, dblquad
from scipy.stats import norm
import numpy as np
import scipy.stats as stats

def likelihood_h1(x, data1, data2, param_X, param_e1, param_e2, f):
	a =  norm.pdf(x, loc=param_X[0], scale=param_X[1])*norm.pdf(data1 - x, loc=param_e1[0], scale=param_e1[1])*norm.pdf(data2 - f(x), loc=param_e2[0], scale=param_e2[1])
	return a

def likelihood_h2(x, data1, param_X, param_e1):
	return norm.pdf(x, loc=param_X[0], scale=param_X[1])*norm.pdf(data1 - x, loc=param_e1[0], scale=param_e1[1])

def calculatePosterior(prior_h1, prior_h2, data, param_X, param_Y, param_e1, param_e2, f1, f2):
	X = data[0]
	Y = data[1]

	likelihood_givenh1_X = quad(likelihood_h1, -np.inf, np.inf, args = (X,Y, param_X, param_e1, param_e2, f1))[0]
	likelihood_givenh1_Y = quad(likelihood_h1, -np.inf, np.inf, args = (Y,X, param_Y, param_e2, param_e1, f2))[0]
	likelihood_givenh1 = (0.5*likelihood_givenh1_X) + (0.5*likelihood_givenh1_Y)
	likelihood_givenh2 = quad(likelihood_h2, -np.inf, np.inf, args = (X,param_X, param_e1))[0]*quad(likelihood_h2, -np.inf, np.inf, args = (Y, param_Y, param_e2))[0]

	posterior_h1 = (prior_h1*likelihood_givenh1)/ ((prior_h1*likelihood_givenh1) + (prior_h2*likelihood_givenh2))
	posterior_h2 = (prior_h2*likelihood_givenh2)/ ((prior_h1*likelihood_givenh1) + (prior_h2*likelihood_givenh2))
	return [posterior_h1, posterior_h2]

def calculateProbability(X,Y, param_X, param_Y, param_e1, param_e2, f1, f2):
	size_X = len(X)
	size_Y = len(Y)
	size = min(size_X, size_Y)

	prob_h1 = 0.5
	prob_h2 = 0.5
	for i in range(size):
		prob_h1, prob_h2 = calculatePosterior(prob_h1, prob_h2, [X[i], Y[i]], param_X, param_Y, param_e1, param_e2, f1, f2)
		print(i, size, prob_h1, prob_h2)
		if(prob_h1==1)or(prob_h2==1):
			break
	return prob_h1

def testDataForLength(association, param_X, param_Y, param_e1, param_e2, peelSize, dataPoints):
	peel1 = np.random.normal(size = peelSize, loc = param_X[0], scale = param_X[1])
	if(not association):
		peel2 = np.random.normal(size = peelSize, loc = param_Y[0], scale = param_Y[1])
	error1 = np.random.normal(size = peelSize, loc = param_e1[0], scale = param_e1[1])
	error2 = np.random.normal(size = peelSize, loc = param_e2[0], scale = param_e2[1])

	peel_1 = np.add(peel1, error1)
	if(association):
		peel_2 = np.add(peel1, error2)
	else:
		peel_2 = np.add(peel2,error2)
	if(association):
		bq, data_bq = calculateProbability(peel_1, peel_2, param_X, param_X, param_e1, param_e2, dataPoints)
	else:
		bq, data_bq = calculateProbability(peel_1, peel_2, param_X, param_Y, param_e1, param_e2, dataPoints)
	correlation = stats.pearsonr(peel_1, peel_2)[0]
	data_correlation = []
	for i in dataPoints:
		data_correlation.append(stats.pearsonr(peel_1[:i], peel_2[:i])[0])
	return [bq, correlation, data_bq, data_correlation]


def testData(association, param_X, param_Y, param_e1, param_e2, peelSize):
	peel1 = np.random.normal(size = peelSize, loc = param_X[0], scale = param_X[1])
	if(not association):
		peel2 = np.random.normal(size = peelSize, loc = param_Y[0], scale = param_Y[1])
	error1 = np.random.normal(size = peelSize, loc = param_e1[0], scale = param_e1[1])
	error2 = np.random.normal(size = peelSize, loc = param_e2[0], scale = param_e2[1])

	peel_1 = np.add(peel1, error1)
	if(association):
		peel_2 = np.add(peel1, error2)
	else:
		peel_2 = np.add(peel2,error2)
	if(association):
		bq = calculateProbability(peel_1, peel_2, param_X, param_X, param_e1, param_e2)
	else:
		bq = calculateProbability(peel_1, peel_2, param_X, param_Y, param_e1, param_e2)
	correlation = stats.pearsonr(peel_1, peel_2)[0]
	return [bq, correlation]


def testData_wrongEstimation(association, param_X, param_Y, param_e1, param_e2, param_X_est, param_Y_est, param_e1_est, param_e2_est, peelSize):
	peel1 = np.random.normal(size = peelSize, loc = param_X[0], scale = param_X[1])
	if(not association):
		peel2 = np.random.normal(size = peelSize, loc = param_Y[0], scale = param_Y[1])
	error1 = np.random.normal(size = peelSize, loc = param_e1[0], scale = param_e1[1])
	error2 = np.random.normal(size = peelSize, loc = param_e2[0], scale = param_e2[1])

	peel_1 = np.add(peel1, error1)
	if(association):
		peel_2 = np.add(peel1, error2)
	else:
		peel_2 = np.add(peel2,error2)
	if(association):
		bq = calculateProbability(peel_1, peel_2, param_X_est, param_X_est, param_e1_est, param_e2_est)
	else:
		bq = calculateProbability(peel_1, peel_2, param_X_est, param_Y_est, param_e1_est, param_e2_est)
	correlation = stats.pearsonr(peel_1, peel_2)[0]
	return [bq, correlation]

