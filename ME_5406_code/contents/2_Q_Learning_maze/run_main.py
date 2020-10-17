"""
Reinforcement learning maze example.
"""
# from algorithms import *
import copy as cp

import numpy as np
# from RL_brain import QLearningTable
from all_algorithms import *
from frozen_lake_env import Frozen_lake
from result_analysis import *
import matplotlib.pyplot as plt

np.random.seed(0)


def run(RL, env, algorithm_name=None, numEpisode=10, max_epsilon=0.98):
    reward_list = []
    num_steps_list = []

    for episode in range(numEpisode):
        print("Episode index :", episode)

        # initial observation
        observation, state = env.reset()

        print("initial observation :", observation)
        print("initial state :", state)

        # epislon_greedy = 0.9 + (max_epsilon - 0.9)/numEpisode * episode
        epislon_greedy = max_epsilon
        print("epislon :", epislon_greedy)

        epi_reward = 0.0
        epi_num_steps = 0
        while epi_num_steps < 100:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(state, epsilon=epislon_greedy)

            print("action ::", action)

            # RL take action and get next observation and reward
            observation_, state_, reward, done = env.step(action)
            epi_reward += reward
            epi_num_steps += 1
            if algorithm_name == "sarsa":
                # RL choose action based on next observation
                action_ = RL.choose_action(state_, epsilon=epislon_greedy)
                # RL learn from this transition
                RL.learn(state, action, reward, state_, action_, done)
            elif algorithm_name == "expected-sarsa":
                # RL learn from this transition
                RL.learn(state, action, reward, state_, done)
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
    algorithm = algorithm_list[3]
    parameters_lr = [1.0, 0.1, 0.01]
    parameters_discount_rate = [1.0, 0.99, 0.9, 0.7, 0.3]
    parameters_epsilon = [0.99, 0.9, 0.85, 0.6, 0.3, 0.1]

    reward_list = []
    value_list = []
    num_steps_list = []

    # para_name = "_epsilon_"
    # parameter_list = parameters_epsilon

    # para_name = "_lr_"
    # parameter_list = parameters_lr
    
    para_name = "_discount_rate_"
    parameter_list = parameters_discount_rate

    run_plot = False
    if run_plot == True:
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for para in parameter_list:
            # set up environment
            # 4 X 4
            # env = Frozen_lake(unit=40,
            #                   grids_height=4, grids_weight=4,
            #                   random_obs=False)

            # 10 X 10
            env = Frozen_lake(unit=40,
                              grids_height=10, grids_weight=10,
                              random_obs=True)

            print("algorithm ：:", algorithm)
            # set algorithm
            if algorithm == "sarsa":
                RL = SarsaTable(actions=list(range(env.n_actions)),
                                states=list(range(env.n_states)),
                                learning_rate=parameters_lr[1],
                                reward_decay=para,
                                e_greedy=parameters_epsilon[0])
            elif algorithm == "expected-sarsa":
                RL = ExpectSarsaTable(actions=list(range(env.n_actions)),
                                      states=list(range(env.n_states)),
                                      learning_rate=parameters_lr[1],
                                      reward_decay=para,
                                      e_greedy=parameters_epsilon[0])
            elif algorithm == "Q-learning":
                RL = QLearningTable(actions=list(range(env.n_actions)),
                                    states=list(range(env.n_states)),
                                    learning_rate=parameters_lr[1],
                                    reward_decay=para,
                                    e_greedy=parameters_epsilon[0])
            else:
                s_a_value = monteCarloNoES(env, numEpisode=20, gamma=1.0, epsilon=0.1)
    
            reward, num_steps, value = run(RL, env, algorithm_name=algorithm,
                                           numEpisode=100,
                                           max_epsilon=parameter_list[0])
            reward_list.append(cp.deepcopy(reward))
            num_steps_list.append(cp.deepcopy(num_steps))
            value_list.append(cp.deepcopy(value))
    
        np.save("./0-data/" + algorithm + para_name + "-lr-value-list.npy", np.array(value_list))
        np.save("./0-data/" + algorithm + para_name + "-lr-reward-list.npy", np.array(reward_list))
        np.save("./0-data/" + algorithm + para_name + "-lr-num-steps-list.npy", np.array(num_steps_list))
    else:
        # plot results
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        value_list = np.load("./0-data/" + algorithm + para_name + "-lr-value-list.npy")
        reward_list = np.load("./0-data/" + algorithm + para_name + "-lr-reward-list.npy")
        num_steps_list = np.load("./0-data/" + algorithm + para_name + "-lr-num-steps-list.npy")

        # print(num_steps_list)
        #
        fig = plt.figure(figsize=(8, 4))
        plt.title(algorithm)
        para_name = 'Lr_'
        for index, reward in enumerate(reward_list):
            plt.plot(np.array(reward), label=para_name + str(index))

        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.savefig("1-figure/" + algorithm + para_name + '_reward.png')
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
        plt.show()

        fig = plt.figure(figsize=(8, 4))
        plt.title(algorithm)
        para_name = 'Lr_'
        for index, reward in enumerate(num_steps_list):
            plt.plot(np.array(reward), label=para_name + str(index))

        plt.xlabel('Episodes')
        plt.ylabel('Episode Steps')
        plt.savefig("1-figure/" + algorithm + para_name + '_steps.png')
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
        plt.show()
