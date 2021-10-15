import itertools

from termcolor import colored

from param import config
import numpy as np
from abc import ABC, abstractmethod
from cpolicies.mpc import take_action_py


class Agent(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def take_action(self, obs_np):
        pass


class BBAAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n
        self.upper = config.bba_reservoir + config.bba_cushion
        self.lower = config.bba_reservoir

    def take_action(self, obs_np):
        buffer_size = obs_np[2 * config.mpc_lookback]
        if buffer_size < self.lower:
            act = 0
        elif buffer_size >= self.upper:
            act = self.act_n - 1
        else:
            ratio = (buffer_size - self.lower) / float(self.upper - self.lower)
            min_chunk = np.min(obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n])
            max_chunk = np.max(obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n])
            bitrate = ratio * (max_chunk - min_chunk) + min_chunk
            act = max([i for i in range(self.act_n) if bitrate >= obs_np[2 * config.mpc_lookback+3+i]])
        return act


class BBAAgentMIX(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n
        self.upper = (config.bba_reservoir + config.bba_cushion) * kwargs['mult']
        self.lower = config.bba_reservoir * kwargs['mult']
        self.ratio_rnd = kwargs['ratio']
        assert 0 <= self.ratio_rnd < 1
        self.rng = np.random.RandomState(config.seed+1)

    def take_action(self, obs_np):
        if self.rng.random() < self.ratio_rnd:
            return self.rng.choice(self.act_n)
        else:
            buffer_size = obs_np[2 * config.mpc_lookback]
            if buffer_size < self.lower:
                act = 0
            elif buffer_size >= self.upper:
                act = self.act_n - 1
            else:
                ratio = (buffer_size - self.lower) / float(self.upper - self.lower)
                min_chunk = np.min(obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n])
                max_chunk = np.max(obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n])
                bitrate = ratio * (max_chunk - min_chunk) + min_chunk
                act = max([i for i in range(self.act_n) if bitrate >= obs_np[2 * config.mpc_lookback+3+i]])
            return act


class RNDAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n
        # random generator, to make runs deterministic, use different seed than env
        self.rng = np.random.RandomState(config.seed+1+kwargs['seed_add'])

    def take_action(self, obs_np):
        return self.rng.choice(self.act_n)


class MPCAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n
        self.rebuf_penalty = kwargs['env'].rebuf_penalty
        self.vid_bit_rate = kwargs['env'].bitrate_map
        self.act_list = [i for i in range(self.act_n)]
        print(colored('Overriding mpc lookahead', 'red'))

    def take_action(self, obs_np):
        next_chunks_len = min(config.mpc_lookahead-1, int(obs_np[2 * config.mpc_lookback + 1]))
        next_chunk_sizes = obs_np[3 + 2 * config.mpc_lookback:3 + 2 * config.mpc_lookback + self.act_n * next_chunks_len]
        past_bandwidths = np.trim_zeros(obs_np[:config.mpc_lookback], 'f')
        if len(past_bandwidths) > 0:
            harmonic_bandwidth = 1 / (1/np.array(past_bandwidths)).mean()
        else:
            harmonic_bandwidth = config.eps
        future_bandwidth = harmonic_bandwidth
        max_reward, best_action = self.recursive_best_mpc(obs_np[2 * config.mpc_lookback], 0, next_chunks_len,
                                                          int(obs_np[2 * config.mpc_lookback + 2]),
                                                          next_chunk_sizes / future_bandwidth)
        return best_action

    def take_action_obsolete(self, obs_np):
        next_chunks_len = min(config.mpc_lookahead, int(obs_np[2 * config.mpc_lookback + 1]))
        next_chunk_sizes = obs_np[3 + 2 * config.mpc_lookback:3 + 2 * config.mpc_lookback + 6 * next_chunks_len]
        chunk_comb_options = [combo for combo in itertools.product(self.act_list, repeat=next_chunks_len)]
        past_bandwidths = np.trim_zeros(obs_np[:config.mpc_lookback], 'f')
        if len(past_bandwidths) > 0:
            harmonic_bandwidth = 1 / (1/np.array(past_bandwidths)).mean()
        else:
            harmonic_bandwidth = config.eps
        future_bandwidth = harmonic_bandwidth
        max_reward = -np.inf
        best_action = 0
        for comb in chunk_comb_options:
            curr_rebuffer_time = 0
            curr_buffer = obs_np[2 * config.mpc_lookback]
            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = int(obs_np[2 * config.mpc_lookback + 2])
            for position in range(next_chunks_len):
                chunk_quality = comb[position]
                # this is MB/MB/s --> seconds
                download_time = next_chunk_sizes[position * self.act_n + chunk_quality] / future_bandwidth
                if curr_buffer < download_time:
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4
                bitrate_sum += self.vid_bit_rate[chunk_quality]
                smoothness_diffs += abs(self.vid_bit_rate[chunk_quality] - self.vid_bit_rate[last_quality])
                last_quality = chunk_quality
            reward = bitrate_sum - (self.rebuf_penalty * curr_rebuffer_time) - smoothness_diffs
            if reward > max_reward:
                max_reward = reward
                best_action = comb[0]
        return best_action

    def recursive_best_mpc(self, curr_buffer, position, recursions_left, last_quality, download_times):
        if recursions_left == 0:
            assert position * self.act_n == len(download_times)
            return 0, 0

        best_reward = -np.inf
        best_act = -1
        for chunk_quality in range(self.act_n):
            reward_act = 0
            buffer_act = curr_buffer
            # this is MB/MB/s --> seconds
            download_time = download_times[position * self.act_n + chunk_quality]
            if buffer_act < download_time:
                reward_act -= self.rebuf_penalty * (download_time - buffer_act)
                buffer_act = 0
            else:
                buffer_act -= download_time
            buffer_act += 4
            reward_act += self.vid_bit_rate[chunk_quality]
            reward_act -= abs(self.vid_bit_rate[chunk_quality] - self.vid_bit_rate[last_quality])
            reward_act += self.recursive_best_mpc(buffer_act, position+1, recursions_left-1, chunk_quality,
                                                  download_times)[0]

            if best_reward < reward_act:
                best_reward = reward_act
                best_act = chunk_quality

        return best_reward, best_act


class CMPCAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n
        self.rebuf_penalty = kwargs['env'].rebuf_penalty
        self.vid_bit_rate = np.array(kwargs['env'].bitrate_map)

    def take_action(self, obs_np):
        return take_action_py(obs_np, self.act_n, self.vid_bit_rate, self.rebuf_penalty, config.mpc_lookback,
                              config.mpc_lookahead, config.eps)


class RateAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n

    def take_action(self, obs_np):
        past_bandwidths = np.trim_zeros(obs_np[:config.mpc_lookback], 'f')
        if len(past_bandwidths) > 0:
            harmonic_bandwidth = 1 / (1/np.array(past_bandwidths)).mean()
        else:
            harmonic_bandwidth = config.eps
        bit_rates = obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n] / 4
        act = max([i for i in range(self.act_n) if harmonic_bandwidth >= bit_rates[i]] + [0])
        return act


class OptimisticRateAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n

    def take_action(self, obs_np):
        past_bandwidths = np.trim_zeros(obs_np[:config.mpc_lookback], 'f')
        if len(past_bandwidths) > 0:
            harmonic_bandwidth = past_bandwidths.max()
        else:
            harmonic_bandwidth = config.eps
        bit_rates = obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n] / 4
        act = max([i for i in range(self.act_n) if harmonic_bandwidth >= bit_rates[i]] + [0])
        return act


class PessimisticRateAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self.act_n = kwargs['env'].action_space.n

    def take_action(self, obs_np):
        past_bandwidths = np.trim_zeros(obs_np[:config.mpc_lookback], 'f')
        if len(past_bandwidths) > 0:
            harmonic_bandwidth = past_bandwidths.min()
        else:
            harmonic_bandwidth = config.eps
        bit_rates = obs_np[2 * config.mpc_lookback+3:2 * config.mpc_lookback+3+self.act_n] / 4
        act = max([i for i in range(self.act_n) if harmonic_bandwidth >= bit_rates[i]] + [0])
        return act


class BolaAgent(Agent):
    MIN_BUF_S = 3
    MAX_BUF_S = 40
    chunk_length = 4

    def __init__(self, **kwargs):
        """
        """
        super().__init__()
        self.act_n = kwargs['env'].action_space.n
        self.size_ladder_bytes = np.array(kwargs['env'].bitrate_map)
        self.utility_ladder = self.utility(self.size_ladder_bytes)

        assert self.size_ladder_bytes[0] < self.size_ladder_bytes[1]
        assert self.utility_ladder[0] < self.utility_ladder[1]
        assert self.MIN_BUF_S < self.MAX_BUF_S

        smallest = {'size': self.size_ladder_bytes[0],
                    'utility': self.utility_ladder[0]}
        second_smallest = {'size': self.size_ladder_bytes[1],
                           'utility': self.utility_ladder[1]}
        largest = {'size': self.size_ladder_bytes[-1],
                   'utility': self.utility_ladder[-1]}

        size_delta = self.size_ladder_bytes[1] - self.size_ladder_bytes[0]
        utility_high = largest['utility']

        size_utility_term = second_smallest['size'] * smallest['utility'] - smallest['size'] * \
            second_smallest['utility']
        gp_nominator = self.MAX_BUF_S * size_utility_term - utility_high * self.MIN_BUF_S * size_delta
        gp_denominator = ((self.MIN_BUF_S - self.MAX_BUF_S) * size_delta)
        self.gp = gp_nominator / gp_denominator
        self.Vp = self.MAX_BUF_S / self.chunk_length / (utility_high + self.gp)

    @staticmethod
    def utility(sizes):
        """

        :param sizes:
        :type sizes: np.ndarray
        :return:
        :rtype: np.ndarray
        """

        return np.log(sizes/sizes[0])

    def take_action(self, obs_np):
        """

        :param obs_np:
        :type obs_np: np.ndarray
        :return: Return the action index, size and ssim
        :rtype: (int, float, float)
        """
        buffer_size = obs_np[2 * config.mpc_lookback]
        buffer_in_chunks = buffer_size / self.chunk_length
        size_arr = obs_np[2 * config.mpc_lookback + 3:2 * config.mpc_lookback + 3 + self.act_n]
        objs = (self.Vp * (self.utility(size_arr) + self.gp) - buffer_in_chunks) / size_arr
        return np.argmax(objs)
