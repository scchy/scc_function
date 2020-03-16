# python 3.6
# author: Scc_hy 
# create date: 2019-11-28
# Function： 将数据导出到excel sheet 中

from openpyxl import load_workbook
import os
import pandas as pd


def data2excel(filename, data_to_write, sheet_name=0):
    """
    Function  
    将数据导出到excel的指定sheet中
    param: filename 文件名 需要输出的文件  
    param: sheet_name 数据输出的sheet名称  
    param: data_to_write vertica导出的pd.DataFrame数据   
    """

    # 判断是否存在文件并创建
    if not os.path.exists(filename):
            dt_tmp = pd.DataFrame()
            dt_tmp.to_excel(filename)

    excelWriter = pd.ExcelWriter(filename, engine='openpyxl')
    book = load_workbook(filename)
    excelWriter.book = book
    data_to_write.to_excel(
        excelWriter, sheet_name=sheet_name, header=True, index=False)
    excelWriter.save()
    excelWriter.close()
    msg = '已经将指定数据输出到[ {} ]的[ {} ]工作薄中'.format(filename, sheet_name)
    return msg
