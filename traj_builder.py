import numpy as np
from tqdm import trange

from env.abr import ABRSimEnv
from policies import Agent


def run_trajectories(env, policy_agent, save_path):
    """

    :param env: The ABR environment, conforming to the standards of Gym API
    :param policy_agent: An ABR policy, needs take_action function to be implemented
    :param save_path: The path to save logged data

    :type env: ABRSimEnv
    :type policy_agent: Agent
    :type save_path: str

    :return:
    """

    # shorthand
    num_traces = len(env.all_traces)        # Number of traces to consider, by default 2000
    len_vid = env.total_num_chunks          # Length of the video

    obs, _ = env.reset()                    # Reset the environment
    size_obs = obs.shape[0]                 # Width of the observation

    # Trajectory to return
    # Each step has obs{of size_obs}, action{of size 1}, reward{of size 1}, next_obs{of size_obs}
    traj = np.empty((num_traces, len_vid, size_obs + 1 + 1 + size_obs + 1))

    for trace_index in trange(num_traces):
        # Go through all bandwidth traces and start from the very first point in the trace
        obs, obs_extended = env.reset(trace_choice=(trace_index, 0))

        # Log episode with that trace:
        for epi_step in range(len_vid):
            # Choose an action through policy
            act = policy_agent.take_action(obs_extended)

            # Take the action
            next_obs, rew, done, info = env.step(act)

            # For MPC, the observation is slightly larger, we'll use the large observation (info['obs_extended']) here.
            # When training RL we'll use the smaller one (first output of step and reset)
            next_obs_extended = info['obs_extended']

            # Save logs
            traj[trace_index][epi_step][:size_obs] = obs
            traj[trace_index][epi_step][size_obs] = act
            traj[trace_index][epi_step][size_obs+1] = rew
            traj[trace_index][epi_step][size_obs+1:2*size_obs+1] = next_obs
            traj[trace_index][epi_step][2*size_obs+2] = float(done)

            # Episode should not finish before video length
            assert not done or epi_step == len_vid-1

            # Going forward: the next state at point t becomes the current state at point t+1
            obs = next_obs
            obs_extended = next_obs_extended

    # Save the logs
    np.save(save_path, traj)
