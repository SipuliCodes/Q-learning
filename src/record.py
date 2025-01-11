import random
import imageio
import numpy as np


def record_video(env, Qtable, out_directory, fps=1):
    images = []
    done = False

    state = env.reset(seed=random.randint(0, 500))
    img = env.render(mode='rgb_array')
    images.append(img)

    while not done:
        action = np.argmax(Qtable[state][:])
        state, reward, done, info = env.step(action)
        img = env.render(mode='rgb_array')
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for i, img in enumerate(images)], duration=1000*1/fps)
    