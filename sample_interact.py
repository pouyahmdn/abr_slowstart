import numpy as np

from env.abr import ABRSimEnv


def run_trajectories():

    # set up environments for workers
    print('Setting up environment..')
    env = ABRSimEnv()

    act_len = env.action_space.n

    # shorthand
    num_traces = 10

    for _ in range(num_traces):
        # Choose specific trace and start from the initial point in the trace
        obs, _ = env.reset()
        done = False
        t = 0

        while not done:
            # choose action through policy
            act = np.random.choice(act_len)

            # take action
            next_obs, rew, done, info = env.step(act)

            print(f'At chunk {t}, the agent took action {act}, and got a reward {rew}')
            print(f'\t\tThe observation was {obs}')

            # next state
            obs = next_obs
            t += 1


if __name__ == '__main__':
    run_trajectories()

