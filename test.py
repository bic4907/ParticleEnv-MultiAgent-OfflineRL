import cv2
import numpy as np

from env.make_env import make_env
import time

env = make_env('simple')


while True:

    key = None
    for _ in range(50):

        actions = []


        if key is None and key == -1:
            actions.append([0, 0])
        else:
            if key == 119:
                actions.append([0, 0.5])
            elif key == 115:
                actions.append([0, -0.5])
            elif key == 97:
                actions.append([-0.5, 0])
            elif key == 100:
                actions.append([0.5, 0])
            else:
                actions.append([0, 0])
        next_state, reward, done, _ = env.step(actions)
        print(reward)
        # print(next_state, reward, done, )
        image = env.render()

        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        time.sleep(0.1)

    env.reset()
