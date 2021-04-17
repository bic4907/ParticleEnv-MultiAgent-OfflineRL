import cv2
import numpy as np

from env.make_env import make_env
import time

env = make_env('simple')
print(env.action_space)
print(env.observation_space)

while True:
    for _ in range(50):

        actions = []
        for _ in range(1):
            ver = float(input())
            hor = float(input())
            actions.append([ver ,hor])
            # actions.append(np.random.uniform(-1, 1, 2))

        next_state, reward, done, _ = env.step(actions)
        print(next_state, reward, done, )
        image = env.render()

        cv2.imshow('image', image)
        cv2.waitKey(1)

    env.reset()
