import numpy as np
import struct
import matplotlib.pyplot as plt


def loadImageSet(filename):
	binfile = open(filename, 'rb')  # 读取二进制文件
	buffers = binfile.read()
	
	head = struct.unpack_from('>IIII', buffers, 0)  # 取前4个整数，返回一个元组
	
	offset = struct.calcsize('>IIII')  # 定位到data开始的位置
	imgNum = head[1]
	width = head[2]
	height = head[3]
	
	bits = imgNum * width * height  # data一共有60000*28*28个像素值
	bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
	
	imgs = struct.unpack_from(bitsString, buffers, offset)  # 取data数据，返回一个元组
	
	binfile.close()
	imgs = np.reshape(imgs, [imgNum, width * height])  # reshape为[60000,784]型数组
	
	return imgs, head


def loadLabelSet(filename):
	binfile = open(filename, 'rb')  # 读二进制文件
	buffers = binfile.read()
	
	head = struct.unpack_from('>II', buffers, 0)  # 取label文件前2个整形数
	
	labelNum = head[1]
	offset = struct.calcsize('>II')  # 定位到label数据开始的位置
	
	numString = '>' + str(labelNum) + "B"  # fmt格式：'>60000B'
	labels = struct.unpack_from(numString, buffers, offset)  # 取label数据
	
	binfile.close()
	labels = np.reshape(labels, [labelNum])  # 转型为列表(一维数组)
	
	return labels, head


def training_image_reconstruction(training_data, index):
	"""
		Display training original images
	"""
	FONT_SIZE = 20
	plt.figure(figsize=(20, 8), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.23, hspace=0.23)
	for i, item in enumerate(index):
		plt.subplot(2, 5, i + 1)
		plt.title('Original_Train_' + str(i), fontsize=FONT_SIZE)
		img_i = training_data[item -1 , :].reshape((28, 28))
		plt.imshow(img_i)
	plt.savefig('original_images_10_new.pdf')


def polynomial_features(data, interaction_bias, degree=2):
	# insert bias 1
	if interaction_bias:
		features = np.insert(data, 0, 1, axis=1)
	else:
		pass
	for d in range(2, degree + 1):
		order_features = []
		for i in range(1 + data.shape[1] * (d-2), features.shape[1]):
			print(i)
			for j in range(i-1, data.shape[1]):
				print(j)
				order_features.append((features[:, i-1] * data[:, j]))
		features = np.append(features, np.array(order_features).reshape(data.shape[0], -1), axis=1)
	return features


def polynomial_features_new(data, interaction_bias, degree=2):
	# insert bias 1
	if interaction_bias:
		features = np.insert(data, 0, 1, axis=1)
	else:
		pass

	for i in range(0, data.shape[1]):
		for j in range(i, data.shape[1]):
			features = np.append(features, np.array(data[:, i] * data[:, j]).reshape(data.shape[0], -1), axis=1)
	
	for i in range(0, data.shape[1]):
		for j in range(i, data.shape[1]):
			for k in range(j, data.shape[1]):
				features = np.append(features, np.array(data[:, i] * data[:, j] * data[:, k]).reshape(data.shape[0], -1), axis=1)
	
	return features


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


def plot_comparision(method='LR'):
	FONT_SIZE = 34
	list = [1, 0.1, 0.001, 0.0001, 0.00001]
	x_list = range(6)
	eta_list = [[0.8771, 0.8771, 0.8771, 0.8771, 0.8771],
				[0.7591, 0.7471, 0.8517, 0.7902, 0.7165],
				[0.7735, 0.8354, 0.8792, 0.8763, 0.808]]
	rate_list = [[0.8466, 0.8466, 0.6881, 0.8735, 0.8371],
				 [0.7658, 0.8725, 0.7607, 0.779, 0.8819],
				 [0.7233, 0.7673, 0.8633, 0.8792, 0.8057]]
	# eta
	# LR_L1 ::: acc[0.7591, 0.7471, 0.8517, 0.7902, 0.7165]
	# LR_L2 ::: acc[0.7735, 0.8354, 0.8792, 0.8763, 0.808]
	# LR_GD ::: acc[0.8771, 0.8771, 0.8771, 0.8771, 0.8771]
	# rate
	# LR_L1 ::: acc [0.7658, 0.8725, 0.7607, 0.779, 0.8819]
	# LR_L2 ::: acc [0.7233, 0.7673, 0.8633, 0.8792, 0.8057]
	# LR_GD ::: acc [0.8466, 0.8466, 0.6881, 0.8735, 0.8371]
	
	plt.figure(figsize=(30, 10), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.23, hspace=0.23)
	plt.title(method, fontsize=FONT_SIZE)
	plt.subplot(1, 2, 1)
	plt.plot(np.array(eta_list[0]), marker='*', markersize=12, linewidth=5, label='LRGD')
	plt.plot(np.array(eta_list[1]), marker='o', markersize=12, linewidth=5, label='LR(L1)')
	plt.plot(np.array(eta_list[2]), marker='s', markersize=12, linewidth=5, label='LR(L2)')
	plt.xlabel('$\eta$', fontsize=FONT_SIZE)
	plt.ylabel('Accuracy', fontsize=FONT_SIZE)
	plt.legend(loc='lower right', fontsize=FONT_SIZE)
	plt.xticks(x_list, list, fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.grid()
	
	plt.subplot(1, 2, 2)
	plt.plot(np.array(rate_list[0]), marker='*', markersize=12, linewidth=5, label='LRGD')
	plt.plot(np.array(rate_list[1]), marker='o', markersize=12, linewidth=5, label='LR(L1)')
	plt.plot(np.array(rate_list[2]), marker='s', markersize=12, linewidth=5, label='LR(L2)')
	plt.xlabel('$\\zeta$', fontsize=FONT_SIZE)
	plt.ylabel('Accuracy', fontsize=FONT_SIZE)
	plt.legend(loc='lower right', fontsize=FONT_SIZE)
	plt.xticks(x_list, list, fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.grid()

	plt.savefig(method + '_accuracy.pdf')
	plt.show()


def plot_comparision_new(method='PR'):
	FONT_SIZE = 34
	list = [1, 0.1, 0.001, 0.0001, 0.00001]
	x_list = range(6)
	eta_list = [[0.9656, 0.9656, 0.9656, 0.9656, 0.9656],
				[0.9658, 0.9644, 0.9656, 0.9656, 0.9656],
				[0.964, 0.9669, 0.9636, 0.9653, 0.9643]]
	rate_list = [[0.9656, 0.9656, 0.9656, 0.9656, 0.9656],
				 [0.9656, 0.9656, 0.9656, 0.9656, 0.9656],
				 [0.9669, 0.9657, 0.9653, 0.9643, 0.9661]]
	
	# PR ::: acc [0.9656, 0.9656, 0.9656, 0.9656, 0.9656]
	# PR_L1 ::: acc [0.9658, 0.9644, 0.9656, 0.9656, 0.9656]
	# PR_L2 ::: acc [0.964, 0.9669, 0.9636, 0.9653, 0.9643]
	# rate
	# PR ::: acc [0.9656, 0.9656, 0.9656, 0.9656, 0.9656]
	# PR_L1 ::: acc [0.9656, 0.9656, 0.9656, 0.9656, 0.9656]
	# PR_L2 ::: acc [0.9669, 0.9657, 0.9653, 0.9643, 0.9661]
	
	plt.figure(figsize=(30, 10), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.95, wspace=0.23, hspace=0.23)
	plt.title(method, fontsize=FONT_SIZE)
	plt.subplot(1, 2, 1)
	plt.plot(np.array(eta_list[0]), marker='*', markersize=12, linewidth=5, label='PRGD')
	plt.plot(np.array(eta_list[1]), marker='o', markersize=12, linewidth=5, label='PR(L1)')
	plt.plot(np.array(eta_list[2]), marker='s', markersize=12, linewidth=5, label='PR(L2)')
	plt.xlabel('$\eta$', fontsize=FONT_SIZE)
	plt.ylabel('Accuracy', fontsize=FONT_SIZE)
	plt.legend(loc='lower right', fontsize=FONT_SIZE)
	plt.xticks(x_list, list, fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.grid()
	
	plt.subplot(1, 2, 2)
	plt.plot(np.array(rate_list[0]), marker='*', markersize=12, linewidth=5, label='PRGD')
	plt.plot(np.array(rate_list[1]), marker='o', markersize=12, linewidth=5, label='PR(L1)')
	plt.plot(np.array(rate_list[2]), marker='s', markersize=12, linewidth=5, label='PR(L2)')
	plt.xlabel('$\\zeta$', fontsize=FONT_SIZE)
	plt.ylabel('Accuracy', fontsize=FONT_SIZE)
	plt.legend(loc='lower right', fontsize=FONT_SIZE)
	plt.xticks(x_list, list, fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.grid()
	
	plt.savefig(method + '_accuracy.pdf')
	plt.show()
	

if __name__ == "__main__":
	
	# file1 = 'train-images-idx3-ubyte'
	# file2 = 'train-labels-idx1-ubyte'
	#
	# file1_test = 't10k-images-idx3-ubyte'
	# file2_test = 't10k-labels-idx1-ubyte'
	#
	# imgs, data_head = loadImageSet(file1)
	# imgs_test, data_head_test = loadImageSet(file1_test)
	#
	# print('data_head:', data_head)
	# print(type(imgs))
	# print('imgs_array:', imgs)
	# print(np.reshape(imgs[1, :], [28, 28]))
	#
	# print('----------我是分割线-----------')
	#
	# labels, labels_head = loadLabelSet(file2)
	# print('labels_head:', labels_head)
	# print(type(labels))
	# print('labels', labels)
	#
	# index_original = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
	# training_image_reconstruction(imgs, index_original)
	
	# index_reconstruct = [4, 3, 2, 19, 5, 9, 12, 1, 62, 8]
	# training_image_reconstruction(imgs_test, index_reconstruct)
	
	# import scipy.io as scio
	#
	# t_images = np.transpose(scio.loadmat('training_images.mat')['data'])
	# print(t_images.shape)
	# training_image_reconstruction(imgs, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	# from mlxtend.data import loadlocal_mnist
	# """ Load data """
	# print('========================= Load Data ================================')
	# training_data, trainig_labels = loadlocal_mnist(images_path='train-images-idx3-ubyte',
	# 												labels_path='train-labels-idx1-ubyte')
	# print('Training data shape ::: ', training_data.shape)
	# print('Labels data ::: ', trainig_labels.shape)
	#
	# test_data, test_labels = loadlocal_mnist(images_path='t10k-images-idx3-ubyte',
	# 										 labels_path='t10k-labels-idx1-ubyte')
	# print('Test data shape ::: ', test_data.shape)
	# print('Labels data shape ::: ', test_labels.shape)
	#
	# """ eigen digit features """
	# print('===================== Eigen Digit Features =============================')
	# training_data_eigen_digit, test_data_eigen_digit = eigen_digit_feature(training_data, test_data)
	# # training_data_eigen_digit = np.insert(training_data_eigen_digit, 0, 1, axis=1)
	# # test_data_eigen_digit = np.insert(test_data_eigen_digit, 0, 1, axis=1)
	# print('Training_data_eigen_digit shape ::: ', training_data_eigen_digit.shape)
	# print('Test_data_eigen_digit shape ::: ', test_data_eigen_digit.shape)
	#
	# from sklearn.preprocessing import PolynomialFeatures
	# poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=True)
	# training_data = poly.fit_transform(training_data_eigen_digit)
	# training_data_new = polynomial_features_new(training_data_eigen_digit, interaction_bias=True, degree=3)
	# print(training_data.shape)
	# print(training_data_new.shape)
	
	# plot_comparision(method='LR')
	plot_comparision_new(method='PR')