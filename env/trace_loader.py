import os

import wget
import zipfile
import numpy as np

import env
from param import config
from scipy.optimize import fsolve


def get_weights(times):
    weights = np.array(times)
    weights[:-1] = weights[1:]-weights[:-1]
    weights[-1] = 1
    return weights


def get_chunk_time(trace, t_idx):
    if t_idx == len(trace[0]) - 1:
        return 1  # bandwidth last for 1 second
    else:
        return trace[0][t_idx + 1] - trace[0][t_idx]


def load_chunk_sizes():
    # bytes of video chunk file at different bitrates

    # source video: "Envivio-Dash3" video H.264/MPEG-4 codec
    # at bitrates in {300,750,1200,1850,2850,4300} kbps

    # original video file:
    # https://github.com/hongzimao/pensieve/tree/master/video_server

    # download video size folder if not existed
    video_folder = env.__path__[0] + '/videos/'
    os.makedirs(video_folder, exist_ok=True)
    if not os.path.exists(video_folder + 'video_sizes.npy'):
        wget.download(
            'https://www.dropbox.com/s/hg8k8qq366y3u0d/video_sizes.npy?dl=1',
            out=video_folder + 'video_sizes.npy')

    chunk_sizes = np.load(video_folder + 'video_sizes.npy')

    return chunk_sizes


def load_traces():
    """
    :rtype: (list of (np.ndarray | list, np.ndarray | list), np.ndarray)
    """
    if config.trace_type == 'real':
        all_traces, all_rtts = load_real_traces()
    elif config.trace_type == 'random':
        all_traces, all_rtts = load_sim_traces_random()
        np.save(config.output_folder + '/traces.npy', [trace[1] for trace in all_traces])
    elif config.trace_type == 'simple':
        all_traces, all_rtts = load_sim_traces_simple()
        np.save(config.output_folder + '/traces.npy', [trace[1] for trace in all_traces])
    elif config.trace_type == 'process':
        all_traces, all_rtts = load_sim_traces_process()
        np.save(config.output_folder + '/traces.npy', [trace[1] for trace in all_traces])
    else:
        raise ValueError('No such trace generation type')

    if not config.disable_slow_start:
        np.save(config.output_folder + '/rtts.npy', all_rtts)

    return all_traces, all_rtts


def load_real_traces():
    """
    :rtype: (list of (np.ndarray | list, np.ndarray | list), np.ndarray)
    """
    # download video size folder if not existed
    trace_folder = env.__path__[0] + '/traces/'

    if not os.path.exists(trace_folder):
        wget.download(
            'https://www.dropbox.com/s/xdlvykz9puhg5xd/cellular_traces.zip?dl=1',
            out=env.__path__[0])
        with zipfile.ZipFile(
             env.__path__[0] + '/cellular_traces.zip', 'r') as zip_f:
            zip_f.extractall(env.__path__[0])

    all_traces = []

    for trace in sorted(os.listdir(trace_folder)):

        all_t = []
        all_bandwidth = []

        with open(trace_folder + trace, 'rb') as f:
            for line in f:
                parse = line.split()
                all_t.append(float(parse[0]))
                all_bandwidth.append(float(parse[1]))

        all_traces.append((all_t, all_bandwidth))

    all_rtts = np.random.RandomState(config.seed).random(size=len(all_traces)) * 180 + 20

    return all_traces, all_rtts


def load_sim_traces_random(length=490):
    """
    :rtype: (list of (np.ndarray | list, np.ndarray | list), np.ndarray)
    """
    all_traces = []
    rng = np.random.RandomState(config.seed)
    for i in range(config.trace_sim_count):
        low_thresh, high_thresh = uniform_thresh(4.5, 0.5, rng)
        all_t = np.arange(length)
        all_bandwidth = rng.random(size=length) * (high_thresh-low_thresh) + low_thresh
        all_traces.append((all_t, all_bandwidth))

    all_rtts = np.random.RandomState(config.seed).random(size=len(all_traces)) * 180 + 20

    return all_traces, all_rtts


def load_sim_traces_simple(length=490):
    """
    :rtype: (list of (np.ndarray | list, np.ndarray | list), np.ndarray)
    """
    all_traces = []
    rng = np.random.RandomState(config.seed)
    for i in range(config.trace_sim_count):
        low_thresh, high_thresh = uniform_thresh(4.5, 0.5, rng)
        repeats = rng.choice(np.arange(4, 30), 2, replace=False)
        low_repeat, high_repeat = np.min(repeats), np.max(repeats)
        assert low_repeat != high_repeat
        all_t = np.arange(length)
        all_bandwidth = np.empty(length)
        j = 0
        bw = 0
        rep = 0
        while j < length:
            if rep > 0:
                all_bandwidth[j] = bw
                j += 1
                rep -= 1
            else:
                bw = rng.random() * (high_thresh-low_thresh) + low_thresh
                rep = rng.randint(low_repeat, high_repeat)
        all_traces.append((all_t, all_bandwidth))

    all_rtts = np.random.RandomState(config.seed).random(size=len(all_traces)) * 180 + 20

    return all_traces, all_rtts


def load_sim_traces_process(length=490):
    """
    :rtype: (list of (np.ndarray | list, np.ndarray | list), np.ndarray)
    """
    rng = np.random.RandomState(config.seed)
    all_traces = []
    for i in range(config.trace_sim_count):
        p_transition = 1 - 1 / rng.randint(30, 100)
        var_coeff = rng.random() * 0.25 + 0.05
        low_thresh, high_thresh = uniform_thresh(4.5, 0.5, rng)
        all_t = np.arange(length)
        all_bandwidth = np.empty(length)
        state = rng.random() * (high_thresh-low_thresh) + low_thresh
        for j in range(length):
            all_bandwidth[j] = np.clip(rng.normal(state, state * var_coeff), low_thresh, high_thresh)
            if rng.random() > p_transition:
                state = doubly_exponential(state, high_thresh, low_thresh, rng)
        all_traces.append((all_t, all_bandwidth))

    all_rtts = np.random.RandomState(config.seed).random(size=len(all_traces)) * 180 + 20

    return all_traces, all_rtts


def doubly_exponential(position, high, low, rng):
    """
    :type position: float
    :type high: float
    :type low: float
    :type rng: np.random.RandomState
    :rtype: float
    """
    lamb = fsolve(lambda la: 1 - np.exp(-la * (high-position)) - np.exp(-la * (position-low)), np.array([0.5]))[0]
    rnd = rng.random()
    if rnd < 1 - np.exp(-lamb * (high-position)):
        return position - np.log(1-rnd)/lamb
    else:
        return position + np.log(rnd) / lamb


def uniform_thresh(high, low, rng):
    """
    :type high: float
    :type low: float
    :type rng: np.random.RandomState
    :rtype: float
    """
    low_thresh, high_thresh = 1, 1
    while (high_thresh-low_thresh) / (high_thresh+low_thresh) < 0.3:
        threshes = rng.random(size=2) * (high - low) + low
        low_thresh, high_thresh = np.min(threshes), np.max(threshes)
    return low_thresh, high_thresh


def sample_trace(all_traces, all_rtts, np_random):
    # weighted random sample based on trace length
    all_p = [len(trace[1]) for trace in all_traces]
    sum_p = float(sum(all_p))
    all_p = [p / sum_p for p in all_p]
    # sample a trace
    trace_idx = np_random.choice(len(all_traces), p=all_p)
    # sample a starting point
    init_t_idx = np_random.choice(len(all_traces[trace_idx][0]))
    # return a trace and the starting t
    return all_traces[trace_idx], all_rtts[trace_idx], init_t_idx
