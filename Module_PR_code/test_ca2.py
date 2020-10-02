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
	mean_data = np.mean(training_data, axis=0)
	training_data = training_data - mean_data
	cov_matrix = training_data.T.dot(training_data)/training_data.shape[0]
	# cal cov_matrix by numpy
	# cov_matrix = np.cov(training_data, rowvar=False, bias=True)
	print('cov_matrix shape ::: ', cov_matrix)
	""" cal eig vector and value """
	eig_val, eig_vec = np.linalg.eig(cov_matrix)
	# column
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
		kernel_matrix = (kernel_matrix + 1) ** gamma
	else:
		print('The default kernel function is linear!!!')
		kernel_matrix = X.dot(A.T)
	return kernel_matrix


def hessian_matrix(training_data, pred_y, training_labels):
	error = pred_y - training_labels
	gradient_w = np.dot(training_data.T, error)
	diagonal_matrix = (pred_y * (1 - pred_y) + 1e-4) * np.eye(training_data.shape[0])
	hess = training_data.T.dot(diagonal_matrix).dot(training_data)
	return hess, gradient_w


def sigmoid(x):
	s = 1. / (1 + np.exp(-x))
	return s


def tanh(x):
	return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))


def linear_regression(training_data, training_labels, test_data, mode=None, eta=0.01):
	"""
		Linear regression (LR)
	"""
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
	
	pred_test_results = test_data.dot(w)
	pred_test_labels = np.argmax(pred_test_results, axis=1)
	print('======== Linear regression regularization %s classification accuracy ========' % mode)
	

def linear_regression_kernel(training_data, training_labels, test_data,
							 mode='gaussian', eta=0.001):
	from sklearn.preprocessing import PolynomialFeatures
	import numpy as np
	from sklearn.metrics import pairwise_distances
	from sklearn.metrics.pairwise import pairwise_kernels
	# [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
	
	# K_t = kernel_function(training_data, test_data, mode=mode, gamma=0.00001)
	# K = kernel_function(training_data, training_data, mode=mode, gamma=0.00001)
	K_t = pairwise_kernels(training_data, test_data, metric='linear')
	K = pairwise_kernels(training_data, training_data, metric='linear')
	pred_test_results = K_t.T.dot(np.linalg.pinv(K + eta * np.identity(training_data.shape[0])).dot(training_labels))
	pred_test_labels = np.argmax(pred_test_results, axis=1)
	print('======== Linear regression kernel %s classification accuracy ========' % mode)


def feature(training_data):
	training_data_new = np.zeros((training_data.shape[0], 3))
	for i in range(training_data.shape[0]):
		training_data_new[i, 0] = training_data[i, 0]
		training_data_new[i, 1] = training_data[i, 1]
		training_data_new[i, 2] = training_data[i, 0] * training_data[i, 1]
		
	return training_data_new


if __name__ == "__main__":
	from sklearn.preprocessing import PolynomialFeatures
	import numpy as np
	from sklearn.metrics import pairwise_distances
	from sklearn.metrics.pairwise import pairwise_kernels
	
	# training_data = np.array([[], []])
# 	# test_data = np.array([])
# 	# training_labels = np.array([1, 1, 1, 1])
# 	# data = np.array([[1.0, 2.], [2.0, 1.0]])
# 	# # poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)
# 	# # print(poly.fit_transform(data))
# 	# # w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
# 	# # pred_test_results = test_data.dot(w)
# 	# # print(pred_test_results)
# 	#
# 	# print(pairwise_kernels(data, data, metric='poly', degree=4))
	
	training_data = np.array([[-1, 1], [-1, 0], [0, 1], [0, 0], [-2, 0], [0, 2], [-3, 0], [0, 3]])
	training_data = feature(training_data)
	print(training_data)
	
	training_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
	# training_labels = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
	
	x = training_data
	# print(np.linalg.inv(np.transpose(x).dot(x)))
	w = np.linalg.pinv(training_data.T.dot(training_data)).dot(training_data.T).dot(training_labels)
	# print(w)
	# x_1 = feature(np.array([[-0.5, 1.5]]))
	# x_2 = feature(np.array([[-1.5, 0.5]]))
	# print(x_1.dot(w))
	# print(x_2.dot(w))
	
	X_3 = np.array([[1, -2], [0, 1], [-1, 0], [1, 0]])
	sort_index, eig_val, eig_vec = cal_cov_matrix(X_3)
	print(sort_index, eig_val, eig_vec)
	eig_vec = eig_vec[:, 0]
	print('eig_vec', eig_vec)
	print(X_3.dot(eig_vec))
	print(eig_vec/(np.sqrt(eig_vec[0]^2 + eig_vec[1]^2)))