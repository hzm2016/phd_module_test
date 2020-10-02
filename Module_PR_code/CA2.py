import numpy as np
import scipy.io as scio
import operator
import time
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
plt.rcParams['font.size'] = 18
FONT_SIZE = 18
NUM_CLASSES = 10


def cal_sub_mean(training_data, test_data):
	"""
		cal_mean of training data
	"""
	mean_data = np.mean(training_data, axis=0)
	print('Mean training shape :::', mean_data.shape)
	subtract_training_data = training_data - mean_data
	print('Subtract training data shape :::', subtract_training_data.shape)
	subtract_test_data = test_data - mean_data
	print('Subtract test data shape', subtract_test_data.shape)
	return mean_data, subtract_training_data, subtract_test_data


def cal_cov_matrix(training_data):
	"""
		cal covariance matrix and eignvalue, eignvector
	"""
	# cov_matrix = np.transpose(training_data).dot(training_data)/(training_data.shape[0] - 1)
	cov_matrix = training_data.T.dot(training_data)
	# cal cov_matrix by numpy
	# cov_matrix = np.cov(training_data, rowvar=False, bias=True)
	print('cov_matrix shape ::: ', cov_matrix.shape)
	""" cal eig vector and value """
	eig_val, eig_vec = np.linalg.eig(cov_matrix)
	# print('val :::', eig_val)
	# print('sorted val :::', np.sort(eig_val))
	""" return the largest max_index eignvalues """
	sort_index = np.argsort(-eig_val)
	eig_val = sorted(eig_val, reverse=True)
	# eig_val = np.sort(-eig_val)
	return sort_index, eig_val, eig_vec


def eigen_digit_feature(training_data, test_data, max_index=30):
	"""
		Features extraction ::: eigen digits
	"""
	mean_data, subtract_training_data, subtract_test_data = cal_sub_mean(training_data, test_data)
	sort_index, eig_val, eig_vec = cal_cov_matrix(subtract_training_data)
	eig_matrix = eig_vec[:, sort_index[:max_index]]
	training_data_eigen_digit = subtract_training_data.dot(eig_matrix)
	test_data_eigen_digit = subtract_test_data.dot(eig_matrix)
	return training_data_eigen_digit, test_data_eigen_digit


def linear_regression(training_data, training_labels, test_data, test_labels,
					  mode=None, solver='gradient', eta=0.01, rate=0.01):
	"""
		Linear regression (LR)
	"""
	params = {
		'rate': rate,
		'maxLoop': 50,
		'eta': eta,
	}
	if solver == 'gradient':
		w = gradient(training_data, training_labels, mode=mode, options=params)
	else:
		if mode == None:
			w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
		elif mode == 'L1':
			w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
		elif mode == 'L2':
			identity_maxtrix = np.identity(training_data.shape[1])
			w = np.linalg.pinv(training_data.T.dot(training_data) +
							   eta * identity_maxtrix).dot(training_data.T).dot(training_labels)
		else:
			w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
	print('weights shape ::: ', w.shape)
	pred_test_results = test_data.dot(w)
	pred_test_labels = np.argmax(pred_test_results, axis=1)
	print('======== Linear regression regularization %s classification accuracy ========' % mode)
	acc = cal_accuracy(test_labels, pred_test_labels)
	print('Classification accuracy test set ::: ', acc)
	return acc


def polynomial_regression(training_data, training_labels, test_data, test_labels,
					  mode=None, degree=2, solver='gradient', eta=0.01, rate=0.01):
	"""
		Polynomial regression(PR)
	"""
	from sklearn.preprocessing import PolynomialFeatures
	poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
	training_data = poly.fit_transform(training_data)
	test_data = poly.fit_transform(test_data)
	# training_data = polynomial_features(training_data, interaction_bias=True, degree=degree)
	# test_data = polynomial_features(test_data, interaction_bias=True, degree=degree)
	params = {
		'rate': rate,
		'maxLoop': 50,
		'eta': eta,
	}
	if solver == 'gradient':
		w = gradient(training_data, training_labels, mode=mode, options=params)
	else:
		if mode == None:
			w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
		elif mode == 'L1':
			w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
		elif mode == 'L2':
			identity_maxtrix = np.identity(training_data.shape[1])
			w = np.linalg.pinv(training_data.T.dot(training_data) +
							   eta * identity_maxtrix).dot(training_data.T).dot(training_labels)
		else:
			w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
	print('weights shape ::: ', w.shape)
	pred_test_results = test_data.dot(w)
	pred_test_labels = np.argmax(pred_test_results, axis=1)
	print('======== Polynomial regression regularization %s classification accuracy ========'%mode)
	acc = cal_accuracy(test_labels, pred_test_labels)
	print('Classification accuracy test set ::: ', acc)
	return acc


def linear_regression_kernel(training_data, training_labels, test_data, test_labels,
							 mode='gaussian', eta=0.001):
	K_t = kernel_function(training_data, test_data, mode=mode, gamma=0.00001)
	K = kernel_function(training_data, training_data, mode=mode, gamma=0.00001)
	pred_test_results = K_t.T.dot(np.linalg.pinv(K + eta * np.identity(training_data.shape[0])).dot(training_labels))
	pred_test_labels = np.argmax(pred_test_results, axis=1)
	print('======== Linear regression kernel %s classification accuracy ========' % mode)
	acc = cal_accuracy(test_labels, pred_test_labels)
	print('Classification accuracy test set ::: ', acc)
	return acc


def polynomial_regression_kernel(training_data, training_labels, test_data, test_labels,
							degree=2, mode='gaussian', eta=0.001):
	from sklearn.preprocessing import PolynomialFeatures
	poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=True)
	training_data = poly.fit_transform(training_data)
	test_data = poly.fit_transform(test_data)
	K_t = kernel_function(training_data, test_data, mode=mode, gamma=0.00001)
	K = kernel_function(training_data, training_data, mode=mode, gamma=0.00001)
	pred_test_results = K_t.T.dot(np.linalg.pinv(K + eta * np.identity(training_data.shape[0])).dot(training_labels))
	pred_test_labels = np.argmax(pred_test_results, axis=1)
	print('======== Polynomial regression kernel %s classification accuracy ========' % mode)
	acc = cal_accuracy(test_labels, pred_test_labels)
	print('Classification accuracy test set ::: ', acc)
	return acc


def gradient(training_data, trainig_labels, mode=None, options=None):
	weight = np.zeros((training_data.shape[1], trainig_labels.shape[1]))
	rate = options.get('rate')
	maxLoop = options.get('maxLoop')
	eta = options.get('eta')
	for i in range(maxLoop):
		pred_results = softmax(training_data.dot(weight))
		if mode == None:
			weight = weight - rate * training_data.T.dot(pred_results - trainig_labels)
		elif mode == 'L2':
			weight = weight - rate * (training_data.T.dot(pred_results - trainig_labels) + eta * weight)
		elif mode == 'L1':
			weight = weight - rate * (training_data.T.dot(pred_results - trainig_labels) + eta * np.sign(weight))
		else:
			weight = weight - rate * training_data.T.dot(pred_results - trainig_labels)
	return weight


def polynomial_features(data, interaction_bias, degree=2):
	if interaction_bias:
		features = np.insert(data, 0, 1, axis=1)
	else:
		features = data
	if degree == 2:
		for i in range(0, data.shape[1]):
			for j in range(i, data.shape[1]):
				features = np.append(features, np.array(data[:, i] * data[:, j]).reshape(data.shape[0], -1), axis=1)
	elif degree == 3:
		for i in range(0, data.shape[1]):
			for j in range(i, data.shape[1]):
				features = np.append(features, np.array(data[:, i] * data[:, j]).reshape(data.shape[0], -1), axis=1)
		for i in range(0, data.shape[1]):
			for j in range(i, data.shape[1]):
				for k in range(j, data.shape[1]):
					features = np.append(features,
										 np.array(data[:, i] * data[:, j] * data[:, k]).reshape(data.shape[0], -1), axis=1)
	else:
		pass
	return features


def softmax(x):
	orig_shape = x.shape
	if len(x.shape) > 1:
		tmp = np.max(x, axis=1)
		x -= tmp.reshape((x.shape[0], 1))
		x = np.exp(x)
		tmp = np.sum(x, axis=1)
		x /= tmp.reshape((x.shape[0], 1))
	else:
		tmp = np.max(x)
		x -= tmp
		x = np.exp(x)
		tmp = np.sum(x)
		x /= tmp
	return x


def labels_to_one_hot(labels_dense, num_classes):
	"""
		Convert class labels from scalars to one-hot vectors.
	"""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot


def cal_accuracy(testlabels, predlabels):
	"""
		cal_accuracy
	"""
	N = 0
	number_of_test = testlabels.shape[0]
	print('number_of_test', number_of_test)
	for i in range(number_of_test):
		if testlabels[i] == predlabels[i]:
			N += 1
	accuracy_rate = N/number_of_test
	return accuracy_rate


def plot_eigenvector(eig_vec, sort_index, max_num):
	"""
		Display the eigenvector
	"""
	plt.figure(figsize=(20, 8), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.23, hspace=0.23)
	for i in range(max_num):
		plt.subplot(2, 5, i + 1)
		plt.title('Eigenvector_' + str(i), fontsize=FONT_SIZE)
		img_i = eig_vec[:, sort_index[i]].reshape((28, 28))
		plt.imshow(img_i)
	plt.savefig('Eigenvector_Top_10.pdf')


def kernel_function(X, A, mode='gaussian', gamma=0.001):
	kernel_matrix = np.zeros((X.shape[0], A.shape[0]))
	if mode == 'gaussian':
		for i in range(X.shape[0]):
			for j in range(A.shape[0]):
				kernel_matrix[i, j] = np.exp(- gamma * (X[i, :].dot(A[j, :])))
	elif mode == 'linear':
		kernel_matrix = X.dot(A.T)
	elif mode == 'poly':
		kernel_matrix = X.dot(A.T)
		kernel_matrix = (kernel_matrix + 1)**gamma
	else:
		print('The default kernel function is linear!!!')
		kernel_matrix = X.dot(A.T)
	return kernel_matrix
	

def image_reconstruction(training_data, test_data,
						 sort_index, eigenvectors,
						 index_original=[], index_reconstruct=[]):
	"""
		Reconstruct original test images
	"""
	plt.figure(figsize=(30, 8), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.23, hspace=0.23)
	Length = 30
	for i, item in enumerate(index_original):
		plt.subplot(2, 10, i + 1)
		plt.title('Training_' + str(item), fontsize=FONT_SIZE)
		img_i = training_data[item-1, :].reshape((28, 28))
		plt.imshow(img_i)
	for i, item in enumerate(index_reconstruct):
		plt.subplot(2, 10, i + 11)
		plt.title('Reconstruct_' + str(item), fontsize=FONT_SIZE)
		img_vec = np.zeros(Length)
		img_i = np.zeros(eigenvectors.shape[1])
		for j in range(Length):
			img_vec[j] = test_data[item-1, :].T.dot(eigenvectors[:, sort_index[j]])
			img_i += img_vec[j] * eigenvectors[:, sort_index[j]]
		img_i = img_i.reshape((28, 28))
		plt.imshow(img_i)
	plt.savefig('reconstruction_images.pdf')
	
	
def training_image_reconstruction(training_data, index):
	"""
		Display training original images
	"""
	plt.figure(figsize=(20, 8), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.23, hspace=0.23)
	for i, item in enumerate(index):
		plt.subplot(2, 5, i + 1)
		plt.title('Original_Train_' + str(i), fontsize=FONT_SIZE)
		img_i = training_data[item, :].reshape((28, 28))
		plt.imshow(img_i)
	plt.savefig('Original_images_10.pdf')


def ablation_study():
	""" regression for classifications """
	algorithms = ['LR', 'PR']
	modes = [None, 'L1', 'L2']
	solvers = [None, 'gradient']
	
	MODE = modes[0]
	algorithm = 'PR'
	params_eta = [1, 0.1, 0.001, 0.0001, 0.00001]
	params_rate = [1, 0.1, 0.001, 0.0001, 0.00001]
	params_acc = []
	if algorithm == 'LR':
		for i, item in enumerate(params_rate):
			acc = linear_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit,
									test_labels,
									mode='L2', solver='gradient', eta=0.001, rate=item)
			params_acc.append(acc)
			print('acc', params_acc)
	elif algorithm == 'PR':
		for i, item in enumerate(params_eta):
			acc = polynomial_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit,
										test_labels,
										mode='L1', degree=3, solver='gradient', eta=item, rate=0.01)
			params_acc.append(acc)
			print('acc', params_acc)
	else:
		linear_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit, test_labels,
						  mode='L2', solver=None, eta=0.001)

	
if __name__ == "__main__":
	""" Load data """
	print('=========================== Load Data ================================')
	training_data, trainig_labels = loadlocal_mnist(images_path='train-images-idx3-ubyte',
													labels_path='train-labels-idx1-ubyte')
	print('Training data shape ::: ', training_data.shape)
	print('Labels data ::: ', trainig_labels.shape)
	""" one hot labels """
	training_labels_one_hot = labels_to_one_hot(trainig_labels, NUM_CLASSES)
	print('One-hot Training Labels shape ::: ', training_labels_one_hot.shape)
	
	test_data, test_labels = loadlocal_mnist(images_path='t10k-images-idx3-ubyte',
											 labels_path='t10k-labels-idx1-ubyte')
	print('Test data shape ::: ', test_data.shape)
	print('Labels data shape ::: ', test_labels.shape)
	
	print('========================= Plot_Eigenvector =============================')
	mean_data, subtract_training_data, subtract_test_data = cal_sub_mean(training_data, test_data)
	sort_index, eig_val, eig_vec = cal_cov_matrix(subtract_training_data)
	# print('Eig_val ::: ', eig_val)
	# print('sort_index ::: ', sort_index)
	# print('eig-val', eig_val)
	plot_eigenvector(eig_vec, sort_index, max_num=10)
	
	print('======================= Images Reconstruction ==========================')
	""" images reconstruction from training test """
	index_original = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	""" images reconstruction from test set """
	index_reconstruct = [4, 3, 2, 19, 5, 9, 12, 1, 62, 8]
	
	image_reconstruction(training_data, test_data,
						 sort_index, eig_vec,
						 index_original, index_reconstruct)
	
	""" eigen digit features """
	print('===================== Eigen Digit Features =============================')
	training_data_eigen_digit, test_data_eigen_digit = eigen_digit_feature(training_data, test_data)
	# training_data_eigen_digit = np.insert(training_data_eigen_digit, 0, 1, axis=1)
	# test_data_eigen_digit = np.insert(test_data_eigen_digit, 0, 1, axis=1)
	print('Training_data_eigen_digit shape ::: ', training_data_eigen_digit.shape)
	print('Test_data_eigen_digit shape ::: ', test_data_eigen_digit.shape)
	
	""" regression for classifications """
	algorithms = ['LR', 'PR', 'LR_kernel']
	modes = [None, 'L1', 'L2']
	solvers = [None, 'gradient']
	
	""" ++++++++++++++++++++++++++++++++++++++++++++ """
	""" parameters are selected here refers to above """
	""" ++++++++++++++++++++++++++++++++++++++++++++ """
	algorithm = 'LR'
	MODE = modes[0]
	solver = solvers[0]
	degree = 2
	if algorithm == 'LR':
		acc = linear_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit, test_labels,
						  mode=MODE, solver=solver, eta=0.0001, rate=0.001)
	elif algorithm == 'PR':
		acc = polynomial_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit, test_labels,
						  mode=MODE, degree=degree, solver=solver, eta=0.0001, rate=0.01)
	elif algorithm == 'LR_kernel':
		linear_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit, test_labels,
						  	  mode='L2', eta=0.001)
	else:
		acc = linear_regression(training_data_eigen_digit, training_labels_one_hot, test_data_eigen_digit, test_labels,
								mode=None, solver=None, eta=0.0001, rate=0.001)