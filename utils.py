# python 3.6
# author: Scc_hy 
# create date: 2019-10-12
# Function： 遍历目录


import sys, os

def lister(root): #对于根目录
    for dirname, subshere, fileshere in os.walk(root):
        print('[ {} ]'.format(dirname))
        for fname in fileshere:
            print(os.path.join(dirname, fname))
