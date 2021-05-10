# python3
# Create date: 2021-05-10
# Author: Scc_hy
# Func: 好用的装饰器

import functools
from concurrent import futures
import logging 
import sys
import time
from functools import wraps
from contextlib import contextmanager


executor = futures.ThreadPoolExecutor(1)

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = executor.submit(func, *args, **kw)
            return future.result(timeout=seconds)
        return wrapper
    return decorator
 



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
