# python 3.6
# author: Scc_hy 
# create date: 2019-10-12
# Function： 遍历目录
# --------------------------
# revise date: 2021-01-31
#         tip: add logger

# -------------------------------

import logging
import sys, os

logging.basicConfig(
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.info('hello')


def lister(root): #对于根目录
    for dirname, subshere, fileshere in os.walk(root):
        print('[ {} ]'.format(dirname))
        for fname in fileshere:
            print(os.path.join(dirname, fname))
