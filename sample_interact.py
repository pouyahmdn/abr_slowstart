import os

import numpy as np

from env.abr import ABRSimEnv
from param import config


def run_trajectories():
    # Create output folder
    os.makedirs(config.output_folder, exist_ok=True)

    # Launch ABR environment
    print('Setting up environment..')
    env = ABRSimEnv()
    # Use this command instead when you want to evaluate your policy
    # env = ABRSimEnv(test_mode=True)
    
    # Shorthand for number of actions
    act_len = env.action_space.n

    # Number of traces we intend to run through, more gives us a better evaluation
    num_traces = 10

    for _ in range(num_traces):
        # Done in reset: Randomly choose a trace and starting point in it
        obs, _ = env.reset()
        done = False
        t = 0

        while not done:
            # Choose an action through random policy
            act = np.random.choice(act_len)

            # Take the action
            next_obs, rew, done, info = env.step(act)

            # Print some statistics
            print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
            print(f'\t\tThe observation was {obs}')

            # Going forward: the next state at point t becomes the current state at point t+1
            obs = next_obs
            t += 1


if __name__ == '__main__':
    run_trajectories()

