"""
Reinforcement Learning Forze Lake example.

"""
import numpy as np
import tkinter as tk
import time
np.random.seed(0)

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Frozen_lake(tk.Tk, object):
    def __init__(self,
                 unit=40,
                 grids_height=4,
                 grids_weight=4,
                 random_obs=False,
                 ):
        super(Frozen_lake, self).__init__()
        self.unit = unit  # pixel of each grid
        self.map_height_size = grids_height
        self.map_weight_size = grids_weight
        self.random_obs = random_obs
        # self.action_space = ['l', 'r', 'u', 'd']
        self.action_space = ['r', 'd', 'l', 'u']
        self.n_actions = len(self.action_space)
        self.n_states = self.map_height_size * self.map_weight_size
        self.title('Froze Lake')
        self.geometry('{0}x{1}'.format(self.map_height_size * self.unit,
                                       self.map_weight_size * self.unit))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self,
                                bg='white',
                                height=self.map_height_size * self.unit,
                                width=self.map_weight_size * self.unit
                                )

        # create grids
        for c in range(0, self.map_weight_size * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.map_height_size * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.map_height_size * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.map_height_size * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create obstacle
        if self.random_obs:
            num_obs = round(self.map_height_size * self.map_weight_size // 4)

            self.hole_list = []
            # generate obstacles ：：：holes
            for i in range(num_obs):
                hole = np.random.randint(low=0, high=self.map_height_size - 1, size=2)
                # print("hole", hole)
                # print("hole list", self.hole_list)
                while (hole.tolist() == [3, 3]) or (hole.tolist() in self.hole_list):
                    hole = np.random.randint(low=0, high=self.map_height_size-1, size=2)
                self.hole_list.append(hole.tolist())

            # create env
            self.hells_list = []
            for hole in self.hole_list:
                hole_center = np.array([hole[0] * self.unit + self.unit//2,
                                        hole[1] * self.unit + self.unit//2])
                self.hell = self.canvas.create_rectangle(
                    hole_center[0] - 15, hole_center[1] - 15,
                    hole_center[0] + 15, hole_center[1] + 15,
                    fill='black')
                self.hells_list.append(self.hell)
        else:
            # set fixed obstacles ：：：holes
            self.hole_list = np.array([[1, 1], [3, 1], [3, 2], [0, 3]])

            # create env
            self.hells_list = []
            for hole in self.hole_list:
                hole_center = np.array([hole[0] * self.unit + self.unit // 2,
                                        hole[1] * self.unit + self.unit // 2])
                self.hell = self.canvas.create_rectangle(
                    hole_center[0] - 15, hole_center[1] - 15,
                    hole_center[0] + 15, hole_center[1] + 15,
                    fill='black')
                self.hells_list.append(self.hell)

        # create frisbee
        oval = np.array([3, 3])
        oval_center = np.array([oval[0] * self.unit + self.unit//2,
                                oval[1] * self.unit + self.unit//2])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create agent
        self.agent = np.array([0, 0])
        start_center = np.array([self.agent[0] * self.unit + self.unit//2,
                                self.agent[1] * self.unit + self.unit//2])
        self.rect = self.canvas.create_rectangle(
            start_center[0] - 15, start_center[1] - 15,
            start_center[0] + 15, start_center[1] + 15,
            fill='red')

        # print([self.canvas.coords(hell) for hell in self.hells_list])
        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)

        # reset agent
        self.agent = np.array([0, 0])
        start_center = np.array([self.agent[0] * self.unit + self.unit // 2,
                                 self.agent[1] * self.unit + self.unit // 2])
        self.rect = self.canvas.create_rectangle(
            start_center[0] - 15, start_center[1] - 15,
            start_center[0] + 15, start_center[1] + 15,
            fill='red')

        # state index
        self.index = self.obtain_index(self.canvas.coords(self.rect))

        return self.canvas.coords(self.rect), self.index

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        # if action == 0:  # left
        #     if s[0] > self.unit:
        #         base_action[0] -= self.unit
        # elif action == 1:   # right
        #     if s[0] < (self.map_weight_size - 1) * self.unit:
        #         base_action[0] += self.unit
        # elif action == 2:   # up
        #     if s[1] > self.unit:
        #         base_action[1] -= self.unit
        # elif action == 3:   # down
        #     if s[1] < (self.map_height_size - 1) * self.unit:
        #         base_action[1] += self.unit

        # new version
        if action == 0:  # right
            if s[0] < (self.map_weight_size - 1) * self.unit:
                base_action[0] += self.unit
        elif action == 1:  # down
            if s[1] < (self.map_height_size - 1) * self.unit:
                base_action[1] += self.unit
        elif action == 2:   # left
            if s[0] > self.unit:
                base_action[0] -= self.unit
        elif action == 3:   # up
            if s[1] > self.unit:
                    base_action[1] -= self.unit

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif s_ in [self.canvas.coords(hell) for hell in self.hells_list]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        self.index = self.obtain_index(self.canvas.coords(self.rect))

        return s_, self.index, reward, done

    def obtain_index(self, state_coords):
        center_0 = (state_coords[0] + state_coords[2])//2
        center_1 = (state_coords[1] + state_coords[3])//2

        index_0 = int(center_0 // self.unit)
        index_1 = int(center_1 // self.unit)

        return index_1 * self.map_height_size + index_0

    def render(self):
        time.sleep(0.1)
        self.update()


if __name__ == "__main__":
    env = Frozen_lake(unit=40,
               grids_height=4, grids_weight=4,
               random_obs=False)
    env.reset()
    env.render()
    time.sleep(1)

    o_, s_index, r, done = env.step(0)
    print("s_ :", s_index)
    print("o_ :", o_)
    print("r :", r)
    env.render()
    time.sleep(1)
    o_, s_index, r, done = env.step(0)
    print("s_ :", s_index)
    print("o_ :", o_)
    print("r :", r)
    env.render()
    time.sleep(1)
    o_, s_index, r, done = env.step(1)
    print("s_ :", s_index)
    print("o_ :", o_)
    print("r :", r)
    env.render()
    time.sleep(1)
    o_, s_index, r, done = env.step(2)
    print("s_ :", s_index)
    print("o_ :", o_)
    print("r :", r)
    env.render()
    time.sleep(1)
    o_, s_index, r, done = env.step(3)
    print("s_ :", s_index)
    print("o_ :", o_)
    print("r :", r)
    env.render()
    time.sleep(1)
    _, state = env.reset()
    env.render()
    time.sleep(1)


