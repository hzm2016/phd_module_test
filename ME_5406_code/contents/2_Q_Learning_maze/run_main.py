"""
Reinforcement learning maze example.

"""
from frozen_lake_env import Frozen_lake
# from RL_brain import QLearningTable
from all_algorithms import *
# from algorithms import *
import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from result_analysis import *
np.random.seed(0)


def run(RL, numEpisode=10, max_epsilon=0.98):
    reward_list = []
    num_steps_list = []
    sarsa = True
    for episode in range(numEpisode):
        print("Episode index :", episode)

        # initial observation
        observation, state = env.reset()

        print("initial observation :", observation)
        print("initial state :", state)

        epislon_greedy = 0.9 + (max_epsilon - 0.9)/numEpisode * episode
        print("epislon :", epislon_greedy)

        epi_reward = 0.0
        epi_num_steps = 0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(state, epsilon=epislon_greedy)

            print("action ::", action)

            # RL take action and get next observation and reward
            observation_, state_, reward, done = env.step(action)
            epi_reward += reward
            epi_num_steps += 1
            if sarsa:
                # RL choose action based on next observation
                action_ = RL.choose_action(state_)
                # RL learn from this transition
                RL.learn(state, action, reward, state_, action_, done)
            else:
                # RL learn from this transition
                RL.learn(state, action, reward, state_, done)

            # print(RL.q_table)

            # swap observation
            observation = observation_
            state = state_

            # break while loop when end of this episode
            if done:
                break

        reward_list.append(cp.deepcopy(epi_reward))
        num_steps_list.append(cp.deepcopy(epi_num_steps))

    # end of game
    print('game over')
    env.destroy()

    return reward_list, num_steps_list, RL.Q_value
    # return reward_list, np.array(RL.q_table.values.tolist())


if __name__ == "__main__":
    # algorithms
    algorithm_list = ["FVMCWOES", "Q-learning", "sarsa", "expected-sarsa"]
    algorithm = algorithm_list[2]
    parameters_lr = [1.0, 0.1, 0.01]
    parameters_epsilon = [0.99, 0.9, 0.85, 0.6]
    para_name = "lr_"
    reward_list = []
    value_list = []
    num_steps_list = []
    for para in parameters_lr:
        # set up environment
        env = Frozen_lake()

        # set algorithm
        if algorithm == "sarsa":
            RL = SarsaTable(actions=list(range(env.n_actions)),
                            states=list(range(env.n_states)),
                            learning_rate=para, reward_decay=0.9, e_greedy=0.9)
        elif algorithm == "Q-learning":
            RL = QLearningTable(actions=list(range(env.n_actions)),
                                states=list(range(env.n_states)),
                                learning_rate=para, reward_decay=0.9, e_greedy=0.9)
        else:
            s_a_value = monteCarloNoES(env, numEpisode=20, gamma=1.0, epsilon=0.1)

        reward, num_steps, value = run(RL, numEpisode=2)
        reward_list.append(cp.deepcopy(reward))
        num_steps_list.append(cp.deepcopy(num_steps))
        value_list.append(cp.deepcopy(value))

    np.save("./0-data/sarsa-lr-value-list.npy", np.array(value_list))
    np.save("./0-data/sarsa-lr-reward-list.npy", np.array(reward_list))
    np.save("./0-data/sarsa-lr-num-steps-list.npy", np.array(num_steps_list))

    # for index, value in enumerate(value_list):
    #     print("value :::", value)
    #     state_value = value.sum(axis=1)
    #     state_value = state_value.reshape((4, 4))
    #     print("state_value", state_value)
    #     # plt_q_table(value=value, name="state_action_value" + str(index))
    #
    #     plt_state_action_arrow_value_table(state_value=state_value, value=value, name="state_action_value_arrow" + str(index))
    #
    #     plt_state_value_table(value=state_value, name="state_value" + str(index))
    #
    # fig = plt.figure()
    # plt.title('Q-learning')
    # for index, reward in enumerate(reward_list):
    #     plt.plot(np.array(reward), label=para_name + str(index))
    #
    # plt.xlabel('Episodes')
    # plt.ylabel('Episode Reward')
    # plt.legend()
    # plt.show()

    # data = np.load("./0-data/value-list.npy")
    # print(data)

    # print(14//4)
    # print(14%4)
    # a = np.ones((2, 4))
    # print(a)
    # a = [a[0, :]]
    # print(a)
    # # a = np.array([[0., -1, 0., 0.]])
    # b = np.insert(a, 0, values=np.array([0, 1, 2, 3]), axis=0)
    # print(b)
    # b = np.random.permutation(b.T)
    # print(b)
    # b = b.T
    # print(b)
    # print(b[1, :].argmax())
    # print(b[0, b[1, :].argmax()])
