import numpy as np
import scipy.io as scio
import operator
import time
import matplotlib.pyplot as plt

data = scio.loadmat('spamData.mat')

# check the data
# print('Training set::', data['Xtrain'].shape)
# print('Training labels::', data['ytrain'].shape)
# print('Testing set::', data['Xtest'].shape)
# print('Testing labels::', data['ytest'].shape)


#===================data processing=====================
"""feature processing"""
def data_processing(testset, mode='log_transformed'):
	if mode == 'binary':
		testset = np.where(testset<0., -1, testset)
		testset = np.where(testset>0., 1, testset)
	else:
		testset = np.log(testset + 0.1)
	return testset

def binary(testset):
	testset = np.where(testset<0.5, 0, testset)
	testset = np.where(testset>0.5, 1, testset)
	return testset

"""classfiy data"""
def classfiy_data(trainingset, traininglabels):
	num_of_samples = trainingset.shape[0]
	data = np.hstack((trainingset, traininglabels))
	data_byclass = {}
	label_byclass = {}
	for i in range(len(data[:, -1])):
		if i in data[:, -1]:
			# print('i', i)
			data_byclass[i] = data[data[:, -1] == i]
			# label_byclass[i] = data[data[:, -1] == i]

	class_name = list(data_byclass.keys())
	# print(class_name)
	# print('data_byclass', data_byclass[1][:, -1])
	num_of_class = len(data_byclass.keys())
	return data_byclass, num_of_samples, num_of_class

#===================results plotting====================
"""cal_accuracy"""
def cal_accuracy(testlabels, predlabels):
	N = 0
	number_of_test = np.array(testlabels).shape[0]
	print('number_of_test', number_of_test)

	for i in range(number_of_test):
		if testlabels[i][0] == predlabels[i]:
			N += 1

	accuracy_rate = N/number_of_test
	return accuracy_rate

"""Plot comparision results"""
def plot_comparision(test_accuracy_list,
					 training_accuracy_list,
					 variable_list,
					 method='KNN',
					 x_label='$\lambda$',
					 y_label='Accuracy Rate'):
	FONT_SIZE = 40
	plt.figure(figsize=(20, 10), dpi=300)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9, wspace=0.23, hspace=0.23)

	plt.subplot(1, 1, 1)
	plt.title(method, fontsize=FONT_SIZE)

	plt.plot(variable_list, test_accuracy_list, linewidth=2.75, label='Train')
	plt.plot(variable_list, training_accuracy_list, linewidth=2.75, label='Test')

	plt.xlabel(x_label, fontsize=FONT_SIZE)
	plt.ylabel(y_label, fontsize=FONT_SIZE)

	plt.legend(loc='best', fontsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.grid()

	# plt.savefig('accuracy.pdf')
	plt.savefig(method + '_error_rate.jpg')
	plt.show()

""" sigmoid """
def sigmoid(x):
    s = 1. / (1 + np.exp(-x))
    return s

""" cal hessian matrix """
def hessian_matrix(trainingset, pred_y):
	diagonal_matrix = (pred_y * (1 - pred_y) + 1e-4) * np.eye(trainingset.shape[0])
	hess = trainingset.T.dot(diagonal_matrix).dot(trainingset)
	return hess

""" cal mean and var for gaussian"""
def cal_mean_var(trainingset):
	X_mean = []
	for i in range(trainingset.shape[1]):
		X_mean.append(np.mean(trainingset[:, i]))

	X_var = []
	for i in range(trainingset.shape[1]):
		X_var.append(np.var(trainingset[:, i]))

	return X_mean, X_var

"""cal guassian priors"""
def cal_gausssian_prob(testset, mean, var):
	gaussian_prob = []
	for a, b, c in zip(testset, mean, var):
		formula1 = np.exp(-(a - b) ** 2 / (2 * c))
		formula2 = 1 / np.sqrt(2 * np.pi * c)
		gaussian_prob.append(formula2 * formula1)
	return gaussian_prob

"""cal feature priors"""
def cal_feature_prior(testset, data_one_class, key, alpha):
	feature_priors = []
	data_one_class = np.concatenate((data_one_class, [testset]), axis=0)
	for i in range(data_one_class.shape[1]):
		N_1 = np.sum(data_one_class[:, i]==1)
		N = data_one_class.shape[0]
		if testset[i] == 1:
			feature_priors.append((N_1 + alpha)/(N + alpha + alpha))
		else:
			feature_priors.append(1 - (N_1 + alpha)/(N + alpha + alpha))
	return feature_priors

#====================== Four methods ==========================
"""KNN"""
def KNN(trainingset, trainglabels, testset, K):
	dataSetSize = trainingset.shape[0]
	datatestset = np.tile(np.array(testset), (dataSetSize, 1, 1)).transpose(1, 0, 2)
	diffMat = datatestset - np.array(trainingset)
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=2)
	distances = sqDistances**0.5
	distances = distances.transpose(1, 0)
	sortedDistIndicies = np.argsort(distances, axis=0)

	testlabels = np.zeros((np.array(testset).shape[0], 1))

	# select the first top K
	for j in range(np.array(testset).shape[0]):
		classCount = {}
		for i in range(K):
			voteIlabel = trainglabels[sortedDistIndicies[i, j]][0]

			# D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None.
			classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

		# sort
		sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
		testlabels[j, :] = sortedClassCount[0][0]

	return testlabels

"""LR"""
def LR(trainingset, traininglabels, testset, lamda=1, n_iterations=40):
	n_features = trainingset.shape[1]
	limit = np.sqrt(1 / n_features)
	weights = np.zeros((n_features, 1))
	bias = 0.
	w = np.insert(weights, 0, bias, axis=0)
	trainingset = np.insert(trainingset, 0, 1, axis=1)
	itr = 0
	while itr <= n_iterations:
		fx = np.dot(trainingset, w)
		pred_y = sigmoid(fx)
		error = (pred_y - traininglabels)
		
		# calculate the gradient
		reg_parameters = np.zeros_like(w)
		reg_parameters[1:] = w[1:]
		gradient_w = np.dot(trainingset.T, error) + lamda * reg_parameters
		
		# calculate the hessian matrix
		reg_matrix = np.eye(w.shape[0])
		reg_matrix[0, 0] = 0.
		hess = hessian_matrix(trainingset, pred_y) + lamda * reg_matrix
		
		# print(hess)
		gradient_newton = np.dot(np.linalg.inv(hess), gradient_w)
		# gradient_newton = np.linalg.solve(hess, gradient_w)
		# print('update_value:::', gradient_newton)
		# update
		w -= gradient_newton
		itr += 1
	# calculate the gradient
	testset = np.insert(testset, 0, 1, axis=1)
	pred_y =  binary(sigmoid(np.dot(testset, w)))
	print('pred_y', pred_y)
	return pred_y.astype(int)

"""Gaussian Naive Bayes"""
def GNB(trainingset, traininglabels, testset):
	data_byclass, num_of_samples, num_of_class = classfiy_data(trainingset, traininglabels)
	prior_prob = []
	X_means = []
	X_vars = []

	# cal priors and mean, var
	for data_one_class in data_byclass.values():
		X_byclass = data_one_class[:, :-1]
		y_byclass = data_one_class[:, -1]
		prior_prob.append((len(X_byclass) + 1) / (num_of_samples + num_of_class))
		mean, var = cal_mean_var(X_byclass)
		X_means.append(mean)
		X_vars.append(var)
	
	posteriori_prob = []
	testlabels = []
	for n in range(np.array(testset).shape[0]):
		for i, j, k in zip(prior_prob, X_means, X_vars):
			gaussian = cal_gausssian_prob(testset[n, :], j, k)
			posteriori_prob.append(np.log(i) + sum(np.log(gaussian)))
		idx = np.argmax(posteriori_prob)
		testlabels.append(idx)
		posteriori_prob = []
	return testlabels

"""Beta-binomial Naive Bayes"""
def BNB(trainingset, traininglabels, testset, alpha):
	data_byclass, num_of_samples, num_of_class = classfiy_data(trainingset, traininglabels)
	prior_prob = []
	X_byclass_list = {}
	for data_one_class in data_byclass.values():
		X_byclass = data_one_class[:, :-1]
		y_byclass = data_one_class[:, -1]
		prior_prob.append((len(X_byclass) + 1) / (num_of_samples + num_of_class))

	posteriori_prob = []
	pred_testlabels = []
	for n in range(np.array(testset).shape[0]):
		for i, data_one_class, key in zip(prior_prob, data_byclass.values(), data_byclass.keys()):
			feature_priors = cal_feature_prior(testset[n, :], data_one_class[:, :-1], key, alpha)
			posteriori_prob.append(np.log(i) + sum(np.log(feature_priors)))
		idx = np.argmax(posteriori_prob)
		pred_testlabels.append([idx])
		posteriori_prob = []
	
	pred_train_labels = []
	for n in range(np.array(trainingset).shape[0]):
		for i, data_one_class, key in zip(prior_prob, data_byclass.values(), data_byclass.keys()):
			feature_priors = cal_feature_prior(trainingset[n, :], data_one_class[:, :-1], key, alpha)
			posteriori_prob.append(np.log(i) + sum(np.log(feature_priors)))
		idx = np.argmax(posteriori_prob)
		pred_train_labels.append([idx])
		posteriori_prob = []
	return pred_train_labels, pred_testlabels


if __name__ == "__main__":
	print(':::::::::::::: Loading Data ::::::::::::')
	testset = data['Xtest']
	testlabels = data['ytest']
	trainingset = data['Xtrain']
	traininglabels = data['ytrain']
	
	print('::::::::::::::start training::::::::::::')
	########################## input method #############################
	methods = ['BNB', 'GNB', 'LR', 'KNN']
	method = 'LR'
	########################## input method #############################
	print('::::::::::::::Method:::::::::::::::::::::::', method)
	
	# =================== Data processing=========================
	if method == 'BNB':
		feature_mode = 'binary'
	else:
		feature_mode = 'log_transformed'
	trainingset = data_processing(trainingset, feature_mode)
	testset = data_processing(testset, feature_mode)
	
	train_error_rate_list = []
	test_error_rate_list = []
	# ====================== training ============================
	if method == 'KNN':
		K_list_1 = np.arange(1, 11, 1)
		K_list_2 = np.arange(15, 105, 5)
		K_list = np.concatenate((K_list_1, K_list_2))
		best_test_rate = 1.
		best_train_rate = 1.
		best_test_K = 0
		best_train_K = 0
		for K in K_list:
			pred_train_labels_1 = KNN(trainingset, traininglabels, trainingset[:2000, :], K)
			pred_train_labels_2 = KNN(trainingset, traininglabels, trainingset[2000:, :], K)
			pred_test_labels = KNN(trainingset, traininglabels, testset, K)
			test_error_rate = 1- cal_accuracy(testlabels, pred_test_labels)
			if best_test_rate > test_error_rate:
				best_test_rate = test_error_rate
				best_test_K = K
				print('best test_error_rate :::', best_test_rate)
				print('best test_error_rate under K = ', best_test_K)
			pred_train_labels = np.concatenate((pred_train_labels_1, pred_train_labels_2) , axis=0)
			print(pred_train_labels.shape)
			train_error_rate = 1 - cal_accuracy(traininglabels, pred_train_labels)
			if best_train_rate > train_error_rate:
				best_train_rate = train_error_rate
				best_train_K = K
				print('best train_error_rate :::', best_train_rate)
				print('best test_error_rate under K = ', best_train_K)
			train_error_rate_list.append(train_error_rate)
			test_error_rate_list.append(test_error_rate)
		np.save(method + '_train_error_rate_list.npy', np.array(train_error_rate_list))
		np.save(method + '_test_error_rate_list.npy', np.array(test_error_rate_list))
	elif method == 'LR':
		lamda_list_1 = np.arange(1, 11, 1)
		lamda_list_2 = np.arange(15, 105, 5)
		lamda_list = np.concatenate((lamda_list_1, lamda_list_2))
		best_test_rate = 1.
		best_train_rate = 1.
		best_test_lambda = 0
		best_train_lamnda = 0
		for lamda in lamda_list:
			pred_test_labels = LR(trainingset, traininglabels, testset, lamda=lamda, n_iterations=20)
			pred_train_labels = LR(trainingset, traininglabels, trainingset, lamda=lamda, n_iterations=20)
			test_error_rate = 1 - cal_accuracy(testlabels, pred_test_labels)
			print('test_error_rate :::', test_error_rate)
			if best_test_rate > test_error_rate:
				best_test_rate = test_error_rate
				best_test_lambda = lamda
				print('best test_error_rate :::', best_test_rate)
				print('best test_error_rate under lambda = ', best_test_lambda)
			train_error_rate = 1 - cal_accuracy(traininglabels, pred_train_labels)
			if best_train_rate > train_error_rate:
				best_train_rate = train_error_rate
				best_train_lamnda = lamda
				print('best train_error_rate :::', best_train_rate)
				print('best test_error_rate under lambda = ', best_train_lamnda)
			print('train_error_rate :::', train_error_rate)
			train_error_rate_list.append(train_error_rate)
			test_error_rate_list.append(test_error_rate)
			test_error_rate = 1 - cal_accuracy(testlabels, pred_test_labels)
		np.save(method + '_train_error_rate_list.npy', np.array(train_error_rate_list))
		np.save(method + '_test_error_rate_list.npy', np.array(test_error_rate_list))
	elif method == 'GNB':
		pred_test_labels = GNB(trainingset, traininglabels, testset)
		pred_train_labels = GNB(trainingset, traininglabels, trainingset)
		test_error_rate = 1 - cal_accuracy(testlabels, pred_test_labels)
		print('test_error_rate :::', test_error_rate)
		train_error_rate = 1 - cal_accuracy(traininglabels, pred_train_labels)
		print('train_error_rate :::', train_error_rate)
		train_error_rate_list.append(train_error_rate)
		test_error_rate_list.append(test_error_rate)
		np.save(method + '_train_error_rate_list.npy', np.array(train_error_rate_list))
		np.save(method + '_test_error_rate_list.npy', np.array(test_error_rate_list))
	elif method == 'BNB':
		alpha_list = np.arange(0, 100.5, 0.5)
		for alpha in alpha_list:
			pred_train_labels, pred_test_labels = BNB(trainingset, traininglabels, testset, alpha)
			test_error_rate = 1- cal_accuracy(testlabels, pred_test_labels)
			test_error_rate_list.append(test_error_rate)
			print('test_error_rate :::', test_error_rate)
			train_error_rate = 1 - cal_accuracy(traininglabels, pred_train_labels)
			print('train_error_rate :::', train_error_rate)
			train_error_rate_list.append(train_error_rate)
		np.save(method + '_train_error_rate_list.npy', np.array(train_error_rate_list))
		np.save(method + '_test_error_rate_list.npy', np.array(test_error_rate_list))
	else:
		pass