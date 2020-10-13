"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
import copy as cp
import matplotlib.pyplot as plt
import numpy as np


def update(RL):
    reward_list = []
    for episode in range(1000):
        # initial observation
        observation = env.reset()
        # print("observation :", observation)
        print("Episode index :", episode)
        epi_reward = 0.0
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            print("action ::", action)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            epi_reward += reward

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))
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

    return reward_list


if __name__ == "__main__":
    # set up environment
    env = Maze()

    # algorithms
    RL = QLearningTable(actions=list(range(env.n_actions)))

    reward_list = update(RL)
    fig = plt.figure()

    # plt.plot(axisX, ordinarySampling, label='Ordinary Importance Sampling')
    plt.plot(np.array(reward_list), label='Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.show()
    # a = np.array([0., 1., 2.])
    # print(a.argmax())
    # env.after(100, update)
    #
    # env.mainloop()
