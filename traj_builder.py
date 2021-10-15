import numpy as np
from tqdm import trange

from env.abr import ABRSimEnv
from policies import Agent


def run_trajectories(env, policy_agent, save_path):
    """

    :param env:
    :param policy_agent:
    :param save_path:

    :type env: ABRSimEnv
    :type policy_agent: Agent
    :type save_path: str

    :return:
    """

    # shorthand
    num_traces = len(env.all_traces)
    len_vid = env.total_num_chunks
    obs, _ = env.reset()
    size_obs = obs.shape[0]

    # trajectory to return, each step has obs{of size_obs}, action{of size 1}, reward{of size 1}, next_obs{of size_obs}
    traj = np.empty((num_traces, len_vid, size_obs + 1 + 1 + size_obs + 1))

    for trace_index in trange(num_traces):
        # Choose specific trace and start from the initial point in the trace
        obs, obs_extended = env.reset(trace_choice=(trace_index, 0))

        for epi_step in range(len_vid):
            # choose action through policy
            act = policy_agent.take_action(obs_extended)

            # take action
            next_obs, rew, done, info = env.step(act)
            next_obs_extended = info['obs_extended']

            # save action
            traj[trace_index][epi_step][:size_obs] = obs
            traj[trace_index][epi_step][size_obs] = act
            traj[trace_index][epi_step][size_obs+1] = rew
            traj[trace_index][epi_step][size_obs+1:2*size_obs+1] = next_obs
            traj[trace_index][epi_step][2*size_obs+2] = float(done)

            # episode should not finish before video length
            assert not done or epi_step == len_vid-1

            # next state
            obs = next_obs
            obs_extended = next_obs_extended

    np.save(save_path, traj)
