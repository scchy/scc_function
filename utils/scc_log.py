
# python3
# Author: Scc_hy
# Create date: 2020-07-11
# Func:  日志， 同时可以用于计时
#        log = Logger()
#        可以用装饰器     
#               @log.timer('xx_func')
#               @log.clock
#               @log.timeout(2)   给任何函数增加超时时间，这个函数在规定时间内还处理不完就可以直接结束任务
#               with log.timer('xx_func'):
#                   script
#       可以行耗时计算
#            log.line_clock('st')
#                   script
#            log.line_clock('ed')
#       可以查看运行耗时
#            log.history
# Revise date: 2021-10-28 
#         Tip: 增加一些装饰器，用于计时
# ===============================================
import logging 
import sys, os
from functools import wraps
from datetime import datetime
from contextlib import contextmanager
from concurrent import futures



class Logger(object):
    def __init__(self, log_open=True, func_open=True):
        """
        可以便捷的开启，关闭日志的输出
        log_open 控制一般日志的输出
        func_open 控制装饰器的输出
        """
        self.log_open = log_open
        self.func_open = func_open
        self.LOGGER = logging.getLogger(__name__)
        self.LOGGER.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
                                            datefmt='%Y-%m-%d %H:%M:%S') 
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.LOGGER.addHandler(handler)
        self._log_db = {}
        self.executor = futures.ThreadPoolExecutor(os.cpu_count() * 4 // 5)

    def debug(self, msg):
        """
        最详细的日志信息，典型应用场景是 问题诊断
        """
        if not self.log_open:
            return None
        self.__clear()
        self.LOGGER.debug(msg)

    def info(self, msg):
        """
        只记录关键节点信息 
        """
        if not self.log_open:
            return None
        self.__clear()
        self.LOGGER.info(msg)
    
    def warning(self, msg):
        """
        当某些不期望的事情发生时记录的信息
        """
        if not self.log_open:
            return None
        self.__clear()
        self.LOGGER.warning(msg)
    
    def error(self, msg):
        """
        更严重的问题导致某些功能不能正常运行时记录的信息
        """
        if not self.log_open:
            return None
        self.__clear()
        self.LOGGER.error(msg)
    
    def critical(self, msg):
        """
        当发生严重错误，导致应用程序不能继续运行时记录的信息
        """
        if not self.log_open:
            return None
        self.__clear()
        self.LOGGER.critical(msg)

    def __clear(self):
        while len(self.LOGGER.handlers) >= 2:
            self.LOGGER.handlers.pop()


    def clock(self, func):
        @wraps(func)
        def clocked(*args, **kwargs):
            st = datetime.now()
            res = func(*args, **kwargs)
            cost_ = datetime.now() - st
            func_name = func.__name__
            self._db_update(func_name, cost_.total_seconds())
            if self.func_open:
                print(f'{func_name} Done ! It cost {cost_}.')
            return res
        return clocked
    
    @contextmanager
    def timer(self, script_description):
        st = datetime.now()
        yield
        cost_ = datetime.now() - st
        self._db_update(script_description, cost_.total_seconds())
        if self.func_open:
            print(f'{script_description} Done ! It cost {cost_}.')
    
    def close(self, log_close=False, func_close=False):
        self.log_open = not log_close
        self.func_open = not func_close

    def line_clock(self, state='start', description=None):
        """
        params state: string state_dict.keys, 建议使用 start end
        """
        state_dict = {
            'start':'start', 'st':'start', 'open':'start',
            'end':'end', 'ed':'end', 'q':'end', 'quit':'end', 'close':'end', 'stop':'end'
        }
        sate_in = state_dict.get(state, 'start')
        cost_time = self.__line_clocks(sate_in)
        if (sate_in == 'end') and description:
            self._db_update(description, cost_time)


    def __line_clocks(self, state):
        out_seconds = 0
        if state == 'start':
            self.tmp_st = datetime.now()
        elif (state == 'end') and hasattr(self, 'tmp_st'):
            out_seconds = (datetime.now() - self.tmp_st).total_seconds()
        elif (state == 'end') and not hasattr(self, 'tmp_st'):
            raise ValueError("Please use 'start' state before this code-line")
        else:
            pass
        return out_seconds

    def _db_update(self, log_desc, cost_time):
        self._log_db[log_desc] = self._log_db.get(log_desc, 0) + cost_time

    @property
    def history(self):
        # normalize
        his_dict = {}
        total = sum(self._log_db.values())
        for k, v in self._log_db.items():
            his_dict[k] = {'cost_time': v, 'cost_time_percent': round(v/(total + 1e-5), 5)}
        return his_dict


    def timeout(self, seconds):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kw):
                future = self.executor.submit(func, *args, **kw)
                return future.result(timeout=seconds)
            return wrapper
        return decorator



log = Logger(log_open=True, func_open=True)


@log.timeout(2)
@log.clock
def add(a, b):
    return a + b


@log.timeout(2)
def aaa(a, b):
    return a * b



for lg, fc in [(i, j) for i in [0, 1] for j in [0, 1]]:
    log.line_clock('st')
    print('=='*15)
    print(f'log_close={lg} func_close={fc}')
    log.close(log_close=lg, func_close=fc)
    log.info('aaa')
    print(add(1, 2))
    print(aaa(1, 2))
    log.line_clock('ed', 'one_loop')
    print(log.history)
