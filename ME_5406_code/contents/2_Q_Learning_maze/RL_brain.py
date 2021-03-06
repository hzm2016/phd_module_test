"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
	def __init__(self,
				 actions,
				 learning_rate=0.5,
				 reward_decay=1.0,
				 e_greedy=0.7):
		self.actions = actions
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions)
		# print("q-table", self.q_table)

	def choose_action(self, observation, epsilon=0.1):
		self.check_state_exist(observation)
		self.epsilon = epsilon

		# epislon-greedy behavior policy to select action
		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.loc[observation, :]
			# print("state_action :", state_action)
			# print("state_action :", state_action.argmax())
			# some actions have same value
			# state_action = state_action.reindex(np.random.permutation(state_action.index))
			# print("state_action :", state_action.index)
			action = state_action.to_numpy().argmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)
		# print("action :", action)
		return action

	def learn(self, s, a, r, s_, done):
		self.check_state_exist(s_)
		# print("q_table :::", self.q_table)
		q_predict = self.q_table.loc[s, a]
		# print("q_predict ::", q_predict)
		if done:
			q_target = r  # next state is terminal
		else:
			q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
			# print("q_target ::", q_target)

		self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

	def check_state_exist(self, state):
		# print("index :", self.q_table.index)
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series(
					[0.0] * len(self.actions),
					index=self.q_table.columns,
					name=state,
				)
			)


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# # off-policy
# class QLearningTable(RL):
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#         super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
#
#     def learn(self, s, a, r, s_):
#         self.check_state_exist(s_)
#         q_predict = self.q_table.ix[s, a]
#         if s_ != 'terminal':
#             q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
#         else:
#             q_target = r  # next state is terminal
#         self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update
#
#
# # on-policy
# class SarsaTable(RL):
#
#     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
#         super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
#
#     def learn(self, s, a, r, s_, a_):
#         self.check_state_exist(s_)
#         q_predict = self.q_table.ix[s, a]
#         if s_ != 'terminal':
#             q_target = r + self.gamma * self.q_table.ix[s_, a_]  # next state is not terminal
#         else:
#             q_target = r  # next state is terminal
#         self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update
