import itertools
import os

from param import config
from policies import BBAAgent, RNDAgent, BBAAgentMIX, CMPCAgent, RateAgent, BolaAgent, OptimisticRateAgent, \
    PessimisticRateAgent
from env import ABRSimEnv
from traj_builder import run_trajectories


def main():
    # Create output folder
    os.makedirs(config.output_folder, exist_ok=True)

    # set up environments for workers
    print('Setting up environment..')
    env = ABRSimEnv()

    if config.add_rnd_policy:
        for multiplier, portion_rnd in itertools.product([1, 2], [0.5]):
            print('Starting BBAMIX..')
            run_trajectories(env, BBAAgentMIX(env=env, mult=multiplier, ratio=portion_rnd),
                             config.output_folder + '/bbamix_X%.1f_RND%d%%_traj.npy' % (multiplier,
                                                                                        int(portion_rnd*100)))

        for i in range(3):
            print('Starting Random..')
            run_trajectories(env, RNDAgent(env=env, seed_add=i), config.output_folder + '/rnd_traj_%d.npy' % i)

    # Running and saving trajectories
    print('Starting BBA..')
    run_trajectories(env, BBAAgent(env=env), config.output_folder + '/bba_traj.npy')

    print('Starting MPC..')
    run_trajectories(env, CMPCAgent(env=env), config.output_folder + '/mpc_traj.npy')

    # Running and saving trajectories
    print('Starting BOLA..')
    run_trajectories(env, BolaAgent(env=env), config.output_folder + '/bola_traj.npy')

    # Running and saving trajectories
    print('Starting Rate based..')
    run_trajectories(env, RateAgent(env=env), config.output_folder + '/rate_traj.npy')

    # Running and saving trajectories
    print('Starting Optimistic Rate based..')
    run_trajectories(env, OptimisticRateAgent(env=env), config.output_folder + '/opt_rate_traj.npy')

    # Running and saving trajectories
    print('Starting Pessimistic Rate based..')
    run_trajectories(env, PessimisticRateAgent(env=env), config.output_folder + '/pess_rate_traj.npy')

    print('DONE')


if __name__ == '__main__':
    main()
