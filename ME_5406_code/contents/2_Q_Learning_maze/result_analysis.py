import seaborn as sns
import matplotlib.pyplot as plt
# matplotlib inline
# from mpl_toolkits.mplot3d import Axes3D
FONT_SIZE = 16
# plt.rcParams['pdf.fonttype'] = FONT_SIZE


def plt_q_table(value, name=None):
	fig = plt.figure(figsize=(10, 10))
	plt.title("State-action Value (Q)", fontsize=FONT_SIZE)
	print("value :::", value)
	h = sns.heatmap(value, square=True, cbar=False, annot=True)
	cb = h.figure.colorbar(h.collections[0])
	cb.ax.tick_params(labelsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.show()


def plt_state_value_table(value, name=None):
	fig = plt.figure(figsize=(10, 10))
	plt.title("State Value (V)", fontsize=FONT_SIZE)
	print("value :::", value)
	h = sns.heatmap(value, square=True, cbar=False, annot=True)
	cb = h.figure.colorbar(h.collections[0])
	cb.ax.tick_params(labelsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.show()


# import numpy as np
# np.random.seed(0)
# import seaborn as sns
#
#
# # sns.set()
# # # sns.set_theme()
# # fig = plt.figure(figsize=(10, 10))
# uniform_data = np.random.rand(10, 12)
# # # sns.heatmap(uniform_data)
# # sns.heatmap(uniform_data, square=True, annot=True)
# # # plt.xticks(fontsize=20) #x轴刻度的字体大小（文本包含在pd_data中了）
# # # plt.yticks(fontsize=20)
# # plt.savefig("heatmap.png")
# # plt.show()
#
#
# fig = plt.figure(figsize=(10, 8))
# h = sns.heatmap(uniform_data, annot=True, linewidths=0.5, cbar=False) #设置不使用其默认自带的colorbar
# cb = h.figure.colorbar(h.collections[0]) #显示colorbar
# cb.ax.tick_params(labelsize=16) #设置colorbar刻度字体大小。
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.show()
