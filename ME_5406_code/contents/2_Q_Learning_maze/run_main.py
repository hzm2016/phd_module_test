"""
Reinforcement learning maze example.

"""
from maze_env import Maze
# from RL_brain import QLearningTable
from all_algorithms import *
# from algorithms import *
import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from result_analysis import *


def run(RL, numEpisode=10):
    reward_list = []
    sarsa = True
    for episode in range(numEpisode):
        print("Episode index :", episode)

        # initial observation
        observation, state = env.reset()

        print("initial observation :", observation)
        print("initial state :", state)

        epislon_greedy = 0.7 + (0.9 - 0.7)/numEpisode * episode
        print("epislon :", epislon_greedy)

        epi_reward = 0.0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(state, epsilon=epislon_greedy)

            print("action ::", action)

            # RL take action and get next observation and reward
            observation_, state_, reward, done = env.step(action)
            epi_reward += reward

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

    # end of game
    print('game over')
    env.destroy()

    return reward_list, RL.Q_value
    # return reward_list, np.array(RL.q_table.values.tolist())


if __name__ == "__main__":
    # algorithms
    # RL = QLearningTable(actions=list(range(env.n_actions)))

    # print(RL.q_table)
    # RL.q_table = RL.q_table.append(
    #     pd.Series(
    #         [1, 2, 3, 4],
    #         index=RL.q_table.columns,
    #         name="test_1",
    #     )
    # )
    # print(RL.q_table)
    # print(RL.q_table.loc["test_1", :])
    # print(RL.q_table.loc["test_1", :].max())
    # RL.q_table.loc["test_1", 0] = RL.q_table.loc["test_1", 0] + 4
    # # print(RL.q_table.loc["test_1", :].to_numpy().argmax())
    # print(RL.q_table.to_numpy())

    parameters = [1.0, 0.1, 0.01]
    para_name = "lr_"
    reward_list = []
    value_list = []
    for para in parameters:
        # set up environment
        env = Maze()
        RL = SarsaTable(actions=list(range(env.n_actions)),
                        states=list(range(env.n_states)),
                        learning_rate=para, reward_decay=0.9, e_greedy=0.9)
        reward, value = run(RL, numEpisode=100)
        reward_list.append(cp.deepcopy(reward))
        value_list.append(cp.deepcopy(value))

    # # value = np.array([[1, 2, 3, 4],
    # #                   [0.0, 0.0, -0.5, 0.0]])
    #
    # # value = pd.to_numeric(RL.q_table.values)
    # print("value :::", RL.q_table.values)

    # value = RL.q_table.values.tolist()
    # print("index :::", RL.q_table.columns)

    # np.save("./0-data/q-learning-value.npy", np.array(value))
    # np.save("./0-data/q-learning-reward.npy", np.array(reward_list))

    for index, value in enumerate(value_list):
        print("value :::", value)
        plt_q_table(value=value, name="state_action_value" + str(index))
        state_value = value.sum(axis=1)
        state_value = state_value.reshape((4, 4))
        print("state_value", state_value)
        plt_state_value_table(value=state_value, name="state_value" + str(index))

    fig = plt.figure()
    plt.title('Q-learning')
    for index, reward in enumerate(reward_list):
        plt.plot(np.array(reward), label=para_name + str(index))
    plt.savefig("1-figure/Q-learning-reward.png")
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.show()

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
