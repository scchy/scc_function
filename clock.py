# python 3
# Author: Scc_hy
# Create date: 2020-07-11

import logging 
import sys
import time
from functools import wraps
from contextlib import contextmanager

def clock(func):
    @wraps(func) 
    def clocked(*args, **kwargs):
        # args_str = ','.join([repr(arg) for arg in args])
        start = time.perf_counter()
        res = func(*args, **kwargs)
        cost_seconds = time.perf_counter() - start
        msg = f'func: {func.__name__}, cost: {cost_seconds:.5f}'
        print(msg)
        return res
    return clocked


@contextmanager
def simple_clock(title): # 类似触发器
    t0 = time.perf_counter()
    yield
    print("{} - done in {:.2f}s".format(title, time.perf_counter() - t0))
