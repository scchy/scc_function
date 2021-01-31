# -*- coding: UTF-8 -*-
# python 3.6
# Author:              Scc_hy
# Create date:         2019-09-19
# Function:            将vertica sql 命令输出到csv
# Concats:             hyscc1994@foxmail.com   
# Tips


import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
import vertica_python

__doc__ = """
            将vertica sql 命令输出到csv
            """

__version__ = '1.0'



account_dict = {


}


conn_info = {
    "host": '...',
    "port": '...',
    "user":'...',
    "password": '...',
    "database": '...',
    "unicode_error": "replace"
}


csv_format_info = {
    'delimiter': ',',
    'quotechar': '"',
    'quoting': True,
    'encoding': 'gb18030'
}

encodings = ('UTF8', 'GB18030', 'GB2312', 'GBK')


class Selection(object):
      """
      负责启动数据库连接、关闭数据连接、查询数据
      例：
            vtc_select = Selection(conn_info)
            vtc_select.__enter__()
            slt_con = vtc_select.select(sql)

            # 写入文件 
            write_select = Write_select(file, slt_con)
            msg = write_select.wirte_2_file(delimiter = ',')

            vtc_select.__exit__()
      """
      def __init__(self, conn_info):
            self.conn_info = conn_info

      def select(self, sql):
            """
            调用Vertica的Python驱动，查询数据，避免内存占用过大，返回一个迭代器
            """
            try:
                  result = self.cur.execute(sql).iterate()
                  return result
            except BaseException as e:
                  logging.error('查询失败，程序即将退出。失败原因如下，请联系{}'.format(__mail__))
                  raise e
                  sys.exit()

      def __enter__(self):
            try:
                  self.conn = vertica_python.connect(**self.conn_info)
                  self.cur = self.conn.cursor()
            except BaseException as e:
                  logging.error('连接失败，程序即将退出。失败原因如下，请联系{}'.format(__mail__))
                  raise e
                  sys.exit()
            return self

      def __exit__(self):
            self.conn.close()



class Write_select():
      """
      将select 出来的内容写入文件
      """
      def __init__(self, file_path, iterate_rows, chuck = 10000):
            self.file_path = file_path
            self.iterate_rows = iterate_rows
            self.chuck = chuck

      def wirte_contents(self, c):
            f = open(self.file_path, 'a+') 
            f.write(c)
            f.close()

      def wirte_2_file(self, delimiter = ','):
            contents = ''
            i = 0 
            chuck_i = 1
            row = ''
            for row in self.iterate_rows:
                  i += 1 
                  try:
                        contents +=  delimiter.join(['{ele}'.format(ele=e) for e in row]) +  '\n'
                  except:
                        contents
                  if i >= self.chuck  * chuck_i:
                        # 每10000 行写入一次
                        self.wirte_contents(contents)
                        chuck_i += 1 
                        contents = ''
                        print('已经写入{}行'.format(i))
            if contents != '':
                  # 当少于10000 或者多余部分 这边写入
                  self.wirte_contents( contents)
            columns = len(row)
            rows = i 
            msg = "导出{}行{}列，到 {} 中".format(rows, columns, self.file_path)
            print(msg)
            return  rows, columns , msg
