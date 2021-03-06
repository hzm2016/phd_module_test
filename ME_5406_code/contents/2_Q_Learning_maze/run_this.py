"""
Reinforcement learning maze example.

"""
from maze_env import Maze
# from RL_brain import QLearningTable
from algorithms import *
import copy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from result_analysis import *


def update(RL, numEpisode=10):
    reward_list = []
    sarsa = True
    for episode in range(numEpisode):
        print("Episode index :", episode)

        # initial observation
        observation, state = env.reset()

        print("initial observation :", observation)
        print("initial state :", state)

        epislon_greedy = 0.1 + (0.9 - 0.1)/numEpisode * episode
        print("epislon :", epislon_greedy)

        epi_reward = 0.0

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation), epsilon=epislon_greedy)

            print("action ::", action)

            # RL take action and get next observation and reward
            observation_, _, reward, done = env.step(action)
            epi_reward += reward

            if sarsa:
                # RL choose action based on next observation
                action_ = RL.choose_action(str(observation_))
                # RL learn from this transition
                RL.learn(str(observation), action, reward, str(observation_), action_, done)
            else:
                # RL learn from this transition
                RL.learn(str(observation), action, reward, str(observation_), done)

            # print(RL.q_table)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

        reward_list.append(cp.deepcopy(epi_reward))

    # end of game
    print('game over')
    env.destroy()

    return reward_list, RL.q_table.to_numpy()


if __name__ == "__main__":
    # set up environment
    env = Maze()

    # algorithms
    # RL = QLearningTable(actions=list(range(env.n_actions)))
    # RL = SarsaTable(actions=list(range(env.n_actions)))

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

    # reward_list, value = update(RL, numEpisode=10)
    # # # print("value :::", value)
    # # #
    # # # value = np.array([[1, 2, 3, 4],
    # # #                   [0.0, 0.0, -0.5, 0.0]])
    # #
    # # # value = pd.to_numeric(RL.q_table.values)
    # # print("value :::", RL.q_table.values)
    # value = RL.q_table.values.tolist()
    # print("index :::", RL.q_table.columns)

    # np.save("./0-data/q-learning-value.npy", np.array(value))
    # np.save("./0-data/q-learning-reward.npy", np.array(reward_list))


    # plt_q_table(value=value)
    # print(np.load("./0-data/q-learning.npy"))

    # fig = plt.figure()
    # plt.plot(np.array(reward_list), label='Q-learning')
    # plt.xlabel('Episodes')
    # plt.ylabel('Episode Reward')
    # plt.legend()
    # plt.show()
    # a = np.array([0., 1., 2.])
    # print(a.argmax())
    # env.after(100, update)
    #
    # env.mainloop()

    a = np.array([[1., 2., 3.], [0., 0., 1.]])
    print(np.sum(a[0, :]))
