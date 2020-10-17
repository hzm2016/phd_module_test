"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import pandas as pd
import time
from result_analysis import *
import copy as cp
import matplotlib.pyplot as plt


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
            # choose best action also with random when some action have same value
            state_action = np.insert([self.Q_value[observation, :]], 0, values=np.array([0, 1, 2, 3]), axis=0)
            state_action = np.random.permutation(state_action.T)
            state_action = state_action.T
            index = state_action[1, :].argmax()
            action = int(state_action[0, index])
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def avaliable_action(self, observation):
        return self.actions

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
            q_target = r  # next state is terminal
        else:
            q_target = r + self.gamma * self.Q_value[s_, a_]  # next state is not terminal
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


# simulate one trajectory
def sample_one_trajectory(env,
                          episode_length,
                          stateActionValues,
                          stateActionPairCount,
                          epsilon=0.1
                          ):
    # reset each episode without exploring start
    _, state = env.reset()

    # select an available action
    action = np.random.choice([0, 1])
    print("init state :", state)
    print("init action :", action)

    time.sleep(1)

    # trajectory
    singleTrajectory = []

    epi_reward = 0.0
    epi_num_steps = 0
    while epi_num_steps < episode_length:
        env.render()
        observation_, state_, reward, done = env.step(action)
        singleTrajectory.append([state, action, reward])
        epi_reward += reward
        epi_num_steps += 1
        state = state_
        action = behaviorPolicy(env, state, stateActionValues, stateActionPairCount, epsilon=epsilon)
        # if done:
        #     break
    return singleTrajectory, epi_reward, epi_num_steps


# epsilon-greedy behavior policy
def behaviorPolicy(env, state, stateActionValues, stateActionPairCount, epsilon=0.1):
    values_ = stateActionValues[state, :] / stateActionPairCount[state, :]
    if np.random.rand() < (1 - epsilon + epsilon / 4):
        # action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # print("value :::", values_)
        # print("action :::", action)
        state_action = np.insert([values_], 0, values=np.array([0, 1, 2, 3]), axis=0)
        state_action = np.random.permutation(state_action.T)
        state_action = state_action.T
        index = state_action[1, :].argmax()
        action = int(state_action[0, index])
        return action
    else:
        return np.random.choice(list(range(env.n_actions)))


# main function for first-visit Monte Carlo control without exploring starts
def monteCarloNoES(env, episode_length=50, numEpisode=10, gamma=1.0, epsilon=0.1):
    reward_list = []
    num_steps_list = []

    # initial state-action-value
    # (n_states, n_actions)
    stateActionValues = np.zeros((env.n_states, env.n_actions))

    # (n_states, n_actions)
    stateActionReturn = np.zeros((env.n_states, env.n_actions))

    # initialze counts to 1 to avoid division by 0
    stateActionPairCount = np.ones((env.n_states, env.n_actions))

    # play for multi episodes
    for episode in range(numEpisode):
        print('episode:', episode)

        trajectory, epi_reward, epi_num_steps = sample_one_trajectory(env, episode_length,
                                                                      stateActionValues, stateActionPairCount, epsilon=epsilon)

        G = 0.0
        # update values of state-action pairs
        for index, one_step in enumerate(reversed(trajectory)):
            state = one_step[0]
            action = one_step[1]
            reward = one_step[2]

            G = gamma * G + reward

            stateActionReturn[state, action] = G

        # judge whether is the first visit and add to state-action value
        for index, one_step in enumerate(trajectory):
            state = one_step[0]
            action = one_step[1]
            if one_step[:2] not in np.array(trajectory)[:, :2][:index]:
                stateActionValues[state, action] += stateActionReturn[state, action]
                stateActionPairCount[state, action] += 1

        reward_list.append(cp.deepcopy(epi_reward))
        num_steps_list.append(cp.deepcopy(epi_num_steps))

    return stateActionValues/stateActionPairCount, reward_list, num_steps_list


if __name__ == "__main__":
    # set up environment
    from frozen_lake_env import Frozen_lake

    parameters_epsilon = [0.99, 0.90, 0.85, 0.60, 0.30, 0.10]
    parameters_list = parameters_epsilon
    reward_list = []
    num_steps_list = []
    for para in parameter_list:
        # set up environment
        env = Frozen_lake(unit=40,
                          grids_height=4, grids_weight=4,
                          random_obs=False)
        value, reward, num_steps = monteCarloNoES(env, numEpisode=100, gamma=1.0, epsilon=para)

    algorithm = "FVMCWOES"
    comparision_performance(value_list=[reward_list],
                            label_list=parameters_epsilon,
                            para_name=r'$\epsilon$',
                            para_name_text='epsilon',
                            y_label_text='Episode Reward',
                            figure_name='reward',
                            algorithm=algorithm)

    comparision_performance(value_list=[num_steps_list],
                            label_list=parameters_epsilon,
                            para_name=r'$\epsilon$',
                            para_name_text='epsilon',
                            y_label_text='Episode Steps',
                            figure_name='steps',
                            algorithm=algorithm)

    print("state_value ::", value)
    state_action = value.sum(axis=1)
    value = np.round(value, 2)
    state_action_value = np.round(np.reshape(state_action, (4, 4)), 2)
    plt_state_value_table(state_action_value, name="state_action_value_" + algorithm)
    plt_state_action_arrow_value_table(state_action_value, value, name="state_action_value_whole_" + algorithm)

    # a = [[1, 0, 0], [2, 1, 0]]
    # print(np.array(a)[:, 0:2][:2])
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
