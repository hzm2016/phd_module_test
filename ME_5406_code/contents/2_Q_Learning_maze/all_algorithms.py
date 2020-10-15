"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, 
                 action_space,
                 state_space,
                 learning_rate=0.01, 
                 reward_decay=0.9, 
                 e_greedy=0.9):
        self.actions = action_space  # a list
        self.states = state_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions)
        self.Q_value = np.zeros((len(self.states), len(self.actions)))

    def choose_action(self, observation, epsilon=0.1):
        self.epsilon = epsilon
        # action selection
        # epislon-greedy behavior policy
        if np.random.rand() < self.epsilon:
            # choose best action
            # state_action = [self.Q_value[observation, :]]
            state_action = np.insert([self.Q_value[observation, :]], 0, values=np.array([0, 1, 2, 3]), axis=0)
            # print("state_action", state_action)
            state_action = np.random.permutation(state_action.T)
            # print("state_action", state_action)
            state_action = state_action.T
            # print("state_action", state_action)
            index = state_action[1, :].argmax()
            action = int(state_action[0, index])
            # print("state_action", action)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, 
                 actions,
                 states,
                 learning_rate=0.01,
                 reward_decay=0.9, 
                 e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, states, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, done):
        # q_predict
        q_predict = self.Q_value[s, a]
        if done:
            q_target = r  # next state is terminal
        else:
            q_target = r + self.gamma * self.Q_value[s_, :].max()  # next state is not terminal

        self.Q_value[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):
    def __init__(self, 
                 actions,
                 states,
                 learning_rate=0.01, 
                 reward_decay=0.9,
                 e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, states, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_, done):
        # q_predict
        q_predict = self.Q_value[s, a]
        if done:
            q_target = r + self.gamma * self.Q_value[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.Q_value[s, a] += self.lr * (q_target - q_predict)  # update

