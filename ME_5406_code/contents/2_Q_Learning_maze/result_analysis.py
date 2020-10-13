import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plt_q_table(value):
	fig = plt.figure(figsize=(10, 10))
	plt.title("Q-value")
	print("value :::", value)
	ax_1 = sns.heatmap(value, cbar=True)

	plt.show()
