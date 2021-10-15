import numpy as np

from env.trace_loader import get_chunk_time


def thr_slow_start(trace, chunk_sizes, cti, rtt, thr_start=1500*2):
    """
    :type trace: tuple of list
    :type chunk_sizes: list of float
    :type cti: int
    :type rtt: float
    :type thr_start: float
    :rtype: (list of float, int)
    """
    # thr_start: bytes/second, Two packets, MTU = 1500 bytes
    thr_end = trace[1][cti] / 8.0 * 1e6  # bytes/second
    len_thr_exp_arr = int(np.ceil(np.log2(thr_end / thr_start)))
    if len_thr_exp_arr <= 0:
        return thr_discrete(trace, chunk_sizes, cti)
    else:
        thr_arr = np.exp2(np.arange(len_thr_exp_arr+1)) * thr_start
        thr_arr[-1] = thr_end
        time_arr = np.ones(len_thr_exp_arr) * rtt / 1000
        cumul_sum_thr = np.cumsum(thr_arr[:-1] * time_arr)
        delay_list = []
        for chunk_size in chunk_sizes:
            index_start = np.where(cumul_sum_thr > chunk_size)[0]
            index_start = len(thr_arr) - 1 if len(index_start) == 0 else index_start[0]
            time_first = 0 if index_start == 0 else rtt / 1000 * index_start
            size_first = 0 if index_start == 0 else cumul_sum_thr[index_start - 1]
            delay = time_first + (chunk_size - size_first) / thr_arr[index_start]
            delay_list.append(delay)
        cti += 1
        if cti == len(trace[1]):
            cti = 0
        return delay_list, cti


def thr_discrete(trace, chunk_sizes, cti):
    """
    :type trace: tuple of list
    :type chunk_sizes: list of float
    :type cti: int
    :rtype: (list of float, int)
    """
    # thr_start: bytes/second, Two packets, MTU = 1500 bytes
    thr_end = trace[1][cti] / 8.0 * 1e6  # bytes/second
    delay_list = []
    for chunk_size in chunk_sizes:
        delay = chunk_size / thr_end
        delay_list.append(delay)
    cti += 1
    if cti == len(trace[1]):
        cti = 0
    return delay_list, cti


def thr_integrate(trace, chunk_sizes, cti, ctl):
    """
    :type trace: tuple of list
    :type chunk_sizes: list of float
    :type cti: int
    :type ctl: float
    :rtype: (list of float, int, float)
    """
    # keep experiencing the network trace
    # until the chunk is downloaded
    delay = 0
    delay_list = []
    while len(chunk_sizes) > 0:  # floating number business
        throughput = trace[1][cti] / 8.0 * 1e6  # bytes/second
        chunk_time_used = min(ctl, chunk_sizes[0] / throughput)
        for i in range(len(chunk_sizes)):
            chunk_sizes[i] -= throughput * chunk_time_used
        ctl -= chunk_time_used
        delay += chunk_time_used

        if ctl == 0:
            cti += 1
            if cti == len(trace[1]):
                cti = 0
            ctl = get_chunk_time(trace, cti)

        if chunk_sizes[0] <= 1e-8:
            chunk_sizes.pop(0)
            delay_list.append(delay)
    return delay_list, cti, ctl
