import matplotlib.pyplot as plt
import numpy as np

K_list_1 = np.arange(1, 11, 1)
K_list_2 = np.arange(15, 105, 5)
K_list = np.concatenate((K_list_1, K_list_2))
alpha_list = np.arange(0, 100.5, 0.5)

lamda_list_1 = np.arange(1, 11, 1)
lamda_list_2 = np.arange(15, 105, 5)
lamda_list = np.concatenate((lamda_list_1, lamda_list_2))

methods = ['BNB', 'GNB', 'LR', 'KNN']
method = 'BNB'

test_accuracy_list = np.load(method + '_test_error_rate_list.npy')
training_accuracy_list = np.load(method + '_train_error_rate_list.npy')


"""Plot comparision results"""
def plot_comparision(test_accuracy_list,
					 training_accuracy_list,
					 variable_list,
					 method='KNN',
					 x_label='$\lambda$',
					 y_label='Accuracy Rate'):
	FONT_SIZE = 40
	plt.figure(figsize=(20, 10), dpi=1000)
	plt.tight_layout(pad=3, w_pad=1., h_pad=0.5)
	plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9, wspace=0.23, hspace=0.23)

	plt.subplot(1, 1, 1)
	plt.title(method, fontsize=FONT_SIZE)

	plt.plot(variable_list, test_accuracy_list, linewidth=5, label='Test')
	plt.plot(variable_list, training_accuracy_list, linewidth=5, label='Train')

	plt.xlabel(x_label, fontsize=FONT_SIZE)
	plt.ylabel(y_label, fontsize=FONT_SIZE)

	plt.legend(loc='best', fontsize=FONT_SIZE)
	list = np.arange(0, 101, 10)
	plt.xticks(list, fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.grid()

	# plt.savefig('accuracy.pdf')
	plt.savefig(method + '_error_rate.pdf')
	plt.show()


if method == 'LR' or method == 'KNN':
	plot_comparision(test_accuracy_list,
					 training_accuracy_list,
					 lamda_list,
					 method=method,
					 x_label=r'$\lambda$',
					 y_label='Error Rate')
elif method == 'BNB':
	plot_comparision(test_accuracy_list,
					 training_accuracy_list,
					 alpha_list,
					 method=method,
					 x_label=r'$\alpha$',
					 y_label='Error Rate')