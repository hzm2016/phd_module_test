import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme(font_scale=2.5)
FONT_SIZE = 20
arrow_width = 0.03


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
					cbar=False, annot=True, annot_kws={'size': 24}, ax=ax)

	cb = h.figure.colorbar(h.collections[0])
	cb.ax.tick_params(labelsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.tight_layout()
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.savefig("1-figure" + "/" + name + ".pdf")
	# plt.show()


def plt_state_action_arrow_value_table(state_value, value, name=None):
	# fig, ax = plt.subplots(figsize=(20, 16))
	# FONT_SIZE = 28
	fig, ax = plt.subplots(figsize=(10, 8))
	FONT_SIZE = 18
	plt.title("State-Action Value (Q)", fontsize=FONT_SIZE)
	print("value :::", value)
	h = sns.heatmap(state_value, square=True, cmap='coolwarm',
					cbar=False, annot=True, annot_kws={'size': 20}, ax=ax)

	arrow_offset = 0.15
	arrow_length = 0.20
	text_offset = 0.35

	length = value.shape[0]
	weight = state_value.shape[0]
	height = state_value.shape[1]
	for i in range(length):
		center = np.array([i%weight + 0.5, i//height + 0.5])

		if i%weight < weight -1:
			# right
			plt.text(text_offset + center[0], -0.15 + center[1], str(value[i, 0]),
					 size=FONT_SIZE,
					 ha='center', va='center', weight="bold", color='black')
			plt.arrow(arrow_offset + center[0], 0. + center[1], arrow_length, 0., width=arrow_width)

		if i//height < height - 1:
			# down
			plt.text(0.2 + center[0], text_offset + center[1], str(value[i, 1]),
					 size=FONT_SIZE,
					 ha='center', va='center', weight="bold", color='black')
			plt.arrow(0. + center[0], arrow_offset + center[1], 0., arrow_length, width=arrow_width)

		if i%weight > 0:
			# left
			plt.text(-text_offset + center[0], -0.15 + center[1], str(value[i, 2]),
					 size=FONT_SIZE,
					 ha='center', va='center', weight="bold", color='black')
			plt.arrow(-arrow_offset + center[0], 0. + center[1], - arrow_length, 0., width=arrow_width)

		if i//height > 0:
			# up
			plt.text(0.2 + center[0], -text_offset + center[1], str(value[i, 3]),
					 size=FONT_SIZE,
					 ha='center', va='center', weight="bold",
					 color='black')
			plt.arrow(0. + center[0], -arrow_offset + center[1], 0., - arrow_length, width=arrow_width)

	cb = h.figure.colorbar(h.collections[0])
	cb.ax.tick_params(labelsize=FONT_SIZE)
	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.tight_layout()
	plt.savefig("1-figure" + "/" + name + ".png")
	plt.savefig("1-figure" + "/" + name + ".pdf")
	# plt.show()


# use to plot reward and steps
def comparision_performance(value_list=None,
							label_list=None,
							para_name=r'$\epsilon$',
							para_name_text='epsilon',
							y_label_text='Episode Steps',
							figure_name='',
							algorithm=''):
	fig = plt.figure(figsize=(10, 5), dpi=600)
	FONT_SIZE = 16
	plt.title(algorithm, fontsize=FONT_SIZE)

	for index, reward in enumerate(value_list):
		plt.plot(np.array(reward), label=para_name + '=' + str(label_list[index]))

	plt.xticks(fontsize=FONT_SIZE)
	plt.yticks(fontsize=FONT_SIZE)
	plt.xlabel('Episodes', fontsize=FONT_SIZE)
	plt.ylabel(y_label_text, fontsize=FONT_SIZE)
	plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), fontsize=FONT_SIZE)
	plt.tight_layout()
	plt.savefig("1-figure/" + algorithm + '_' + para_name_text + '_' + figure_name + '.pdf')


# use to plot reward and steps
def comparision_all_algorithms_performance(value_list=None,
							label_list=None,
							para_name=r'$\epsilon$',
							para_name_text='epsilon',
							y_label_text='Episode Steps',
							figure_name='',
							algorithm=''):
	fig = plt.figure(figsize=(15, 7), dpi=600)
	plt.title(algorithm)

	for index, reward in enumerate(value_list):
		plt.plot(np.array(reward)[:100], label=para_name + str(label_list[index]))

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