import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# matplotlib inline
# from mpl_toolkits.mplot3d import Axes3D
sns.set_theme(font_scale=2.0) # font_scale=1.5
FONT_SIZE = 16
arrow_width = 0.03
# plt.rcParams['pdf.fonttype'] = FONT_SIZE


def plt_q_table(value, name=None):
	fig, ax = plt.subplots(figsize=(4, 8))
	plt.title("State-action Value (Q)", fontsize=FONT_SIZE)
	print("value :::", value)
	h = sns.heatmap(value, square=True, cmap='coolwarm',
					cbar=False, annot=True, annot_kws={'size': 16}, ax=ax)
	cb = h.figure.colorbar(h.collections[0])
	cb.ax.tick_params(labelsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.savefig("1-figure" + "/" + name + ".pdf")
	# plt.show()


def plt_state_value_table(value, name=None):
	fig, ax = plt.subplots(figsize=(10, 8))
	plt.title("State Value (V)", fontsize=FONT_SIZE)
	# print("value :::", value)
	h = sns.heatmap(value, square=True, cmap='coolwarm',
					cbar=False, annot=True, annot_kws={'size': 16}, ax=ax)

	# # plot q-value and action selection
	# center_list = [[1.5, 1.5], [1.5, 2.5]]
	# 	# np.array([1.5, 1.5])
	# arrow_offset = 0.1
	# arrow_length = 0.25
	# text_offset = 0.35
	#
	# for center in center_list:
	# 	# right
	# 	plt.text(text_offset + center[0], -0.1 + center[1], "0.1", size=FONT_SIZE, ha='center', va='center',
	# 			 color='darkred')
	# 	plt.arrow(arrow_offset + center[0], 0. + center[1], arrow_length, 0., width=arrow_width)
	#
	# 	# down
	# 	plt.text(0.15 + center[0], text_offset + center[1], "0.2", size=FONT_SIZE, ha='center', va='center',
	# 			 color='darkred')
	# 	plt.arrow(0. + center[0], arrow_offset + center[1], 0., arrow_length, width=arrow_width)
	#
	# 	# left
	# 	plt.text(-text_offset + center[0], -0.1 + center[1], "0.3", size=FONT_SIZE, ha='center', va='center',
	# 			 color='darkred')
	# 	plt.arrow(-arrow_offset + center[0], 0. + center[1], - arrow_length, 0., width=arrow_width)
	#
	# 	# up
	# 	plt.text(0.15 + center[0], -text_offset + center[1], "0.2", size=FONT_SIZE, ha='center', va='center',
	# 			 color='darkred')
	# 	plt.arrow(0. + center[0], -arrow_offset + center[1], 0., - arrow_length, width=arrow_width)

	cb = h.figure.colorbar(h.collections[0])
	cb.ax.tick_params(labelsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.tight_layout()
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.savefig("1-figure" + "/" + name + ".pdf")
	# plt.show()


def plt_state_action_arrow_value_table(state_value, value, name=None):
	fig, ax = plt.subplots(figsize=(10, 8))
	plt.title("State-Action Value (Q)", fontsize=FONT_SIZE)
	print("value :::", value)
	h = sns.heatmap(state_value, square=True, cmap='coolwarm',
					cbar=True, annot=True, annot_kws={'size': 16}, ax=ax)

	# plot q-value and action selection
	# center_list = [[1.5, 1.5], [1.5, 2.5]]
		# np.array([1.5, 1.5])

	arrow_offset = 0.1
	arrow_length = 0.25
	text_offset = 0.35

	for i in range(16):
		center = np.array([i%4 + 0.5, i//4 + 0.5])
		# print("center ::", center)

		# right
		plt.text(text_offset + center[0], -0.1 + center[1], str(value[i, 0]),
				 size=FONT_SIZE,
				 ha='center', va='center', weight="bold", color='black')
		plt.arrow(arrow_offset + center[0], 0. + center[1], arrow_length, 0., width=arrow_width)

		# down
		plt.text(0.15 + center[0], text_offset + center[1], str(value[i, 1]),
				 size=FONT_SIZE,
				 ha='center', va='center', weight="bold", color='black')
		plt.arrow(0. + center[0], arrow_offset + center[1], 0., arrow_length, width=arrow_width)

		# left
		plt.text(-text_offset + center[0], -0.1 + center[1], str(value[i, 2]),
				 size=FONT_SIZE,
				 ha='center', va='center', weight="bold", color='black')
		plt.arrow(-arrow_offset + center[0], 0. + center[1], - arrow_length, 0., width=arrow_width)

		# up
		plt.text(0.15 + center[0], -text_offset + center[1], str(value[i, 3]),
				 size=FONT_SIZE,
				 ha='center', va='center', weight="bold",
				 color='black')
		plt.arrow(0. + center[0], -arrow_offset + center[1], 0., - arrow_length, width=arrow_width)

	# cb = h.figure.colorbar(h.collections[0])
	# cb.ax.tick_params(labelsize=FONT_SIZE)
	# plt.xticks(fontsize=FONT_SIZE)
	# plt.yticks(fontsize=FONT_SIZE)
	plt.tight_layout()
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.savefig("1-figure" + "/" + name + ".pdf")
	# plt.show()


# use to plot reward and steps
def comparision_performance(value_list=None, label_list=None,
							para_name=r'$\epsilon$',
							para_name_text='epsilon',
							y_label_text='Episode Steps',
							figure_name='',
							algorithm=''):
	fig = plt.figure(figsize=(10, 5), dpi=600)
	plt.title(algorithm)

	for index, reward in enumerate(value_list):
		plt.plot(np.array(reward)[:100], label=para_name + '=' + str(label_list[index]))

	plt.xlabel('Episodes')
	plt.ylabel(y_label_text)
	plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
	plt.tight_layout()
	plt.savefig("1-figure/" + algorithm + '_' + para_name_text + '_' + figure_name + '.pdf')


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

# # plot results
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# value_list = np.load("./0-data/" + algorithm + para_name + "-lr-value-list.npy")
# reward_list = np.load("./0-data/" + algorithm + para_name + "-lr-reward-list.npy")
# num_steps_list = np.load("./0-data/" + algorithm + para_name + "-lr-num-steps-list.npy")
# #
# # # print(num_steps_list)
# #
# fig = plt.figure(figsize=(10, 5), dpi=600)
# plt.title(algorithm)
# # para_name = 'Lr_'
# para_name = r'$\epsilon$'
# for index, reward in enumerate(reward_list):
# 	plt.plot(np.array(reward)[:100], label=para_name + '=' + str(parameter_list[index]))
# 	print(para_name + str(index))
#
# para_name = 'epsilon'
# plt.xlabel('Episodes')
# plt.ylabel('Episode Reward')
# plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
# plt.tight_layout()
# plt.savefig("1-figure/" + algorithm + '_' + para_name + '_reward.pdf')
# # plt.show()
#
# fig = plt.figure(figsize=(10, 5), dpi=600)
# plt.title(algorithm)
# # para_name = 'Lr_'
# para_name = r'$\epsilon$'
# for index, reward in enumerate(num_steps_list):
# 	plt.plot(np.array(reward)[:100], label=para_name + '=' + str(parameter_list[index]))
#
# para_name = 'epsilon'
# plt.xlabel('Episodes')
# plt.ylabel('Episode Steps')
# plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
# plt.tight_layout()
# plt.savefig("1-figure/" + algorithm + '_' + para_name + '_steps.pdf')
# # plt.show()

# for index, value in enumerate(value_list[0]):
# value = value_list[0]
# print("value ::", value)
# state_action = value.sum(axis=1)
# value = np.round(value, 2)
# state_action_value = np.round(np.reshape(state_action, (4, 4)), 2)
# plt_q_table(value, name="Q-value")
# plt_state_value_table(state_action_value, name="state_action_value")
# plt_state_action_arrow_value_table(state_action_value, value, name="state_action_value_whole")