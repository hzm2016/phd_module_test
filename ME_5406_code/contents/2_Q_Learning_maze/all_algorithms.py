"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import pandas as pd
import time
from result_analysis import *


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


# off-policy
class ExpectSarsaTable(RL):
    def __init__(self,
                 actions,
                 states,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9):
        super(ExpectSarsaTable, self).__init__(actions, states, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, done):
        # q_predict
        q_predict = self.Q_value[s, a]
        if done:
            # next state is not terminal
            q_target = r + self.gamma * np.sum(self.Q_value[s_, :])
        else:
            # next state is terminal
            q_target = r
        # update based on TD
        self.Q_value[s, a] += self.lr * (q_target - q_predict)


# First Visit MC_without_ES
def sample_one_trajectory(env,
                          stateActionValues,
                          stateActionPairCount,
                          epsilon=0.1
                          ):
    # reset each episode without exploring start
    _, state = env.reset()

    # select an available action
    action = np.random.choice([0, 1])
    # print("init state :", state)
    # print("init action :", action)

    time.sleep(1)

    # trajectory
    singleTrajectory = []

    while True:
        env.render()
        observation_, state_, reward, done = env.step(action)
        singleTrajectory.append([state, action, reward])
        # print("action", action)
        # print("state", state)
        state = state_
        action = behaviorPolicy(state, stateActionValues, stateActionPairCount, epsilon=epsilon)
        if done:
            break
    return singleTrajectory


# define epsilon-greedy behavior policy
def behaviorPolicy(state, stateActionValues, stateActionPairCount, epsilon=0.1):
    values_ = stateActionValues[state, :] / stateActionPairCount[state, :]
    if np.random.rand() < (1 - epsilon + epsilon / 4):
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
    else:
        return np.random.choice(list(range(env.n_actions)))


def monteCarloNoES(env, numEpisode=10, gamma=1.0, epsilon=0.1):
    # initial state-action-value
    # (n_states, n_actions)
    stateActionValues = np.zeros((env.n_states, env.n_actions))

    # initialze counts to 1 to avoid division by 0
    stateActionPairCount = np.ones((env.n_states, env.n_actions))

    # play for multi episodes
    for episode in range(numEpisode):
        # if episode % 10 == 0:
        print('episode:', episode)

        trajectory = sample_one_trajectory(env, stateActionValues, stateActionPairCount, epsilon=epsilon)

        G = 0.0
        for index, one_step in enumerate(reversed(trajectory)):
            # update values of state-action pairs
            state = one_step[0]
            action = one_step[1]
            reward = one_step[2]

            G = gamma * G + reward

            trajectory = list(reversed(trajectory))
            # judge whether is the first visit
            if state not in np.array(trajectory)[:, 0][:index]:
                print("")
                stateActionValues[state, action] += G
                stateActionPairCount[state, action] += 1

    return stateActionValues / stateActionPairCount


if __name__ == "__main__":
    # set up environment
    from frozen_lake_env import Frozen_lake
    env = Frozen_lake()
    # _, state = env.reset()
    s_a_distribution = monteCarloNoES(env, numEpisode=20, gamma=1.0, epsilon=0.1)
    print("state_value ::", s_a_distribution)
    plt_q_table(s_a_distribution, name="state_action_value_" + str(0))
    # s_distribution = np.sum(s_a_distribution, axis=1)
    # fig = plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    # ax_1 = sns.heatmap(s_a_distribution, cbar=True)
    # print("state_value ::", s_distribution)
    # plt.subplot(122)
    # ax_2 = sns.heatmap(s_distribution.reshape((4, 4)), cbar=True)
    # # heatmap = plt.pcolor(value_distribution, cmap='RdBu')
    # plt.show()

    # a = [[1], [2], [3]]
    # a = list(reversed(a))
    # for i, value in enumerate(a):
    #     print("a", a)
    #     print(value)
    #     print(i)
    #     # b = list(reversed(a))
    #     # print(b[:i])
    #     print(a[:i])
