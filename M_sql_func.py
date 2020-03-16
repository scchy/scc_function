# python 3.6
# author: Scc_hy
# create date: 2020-02-14
# 


import pandas as pd
import numpy as np


local_lst = [chr(i) for i in range(ord('A'), ord('L'))]



def sql2df(vtc, sql):
    slt_conn = vtc.execute(sql).fetchall()
    df = pd.DataFrame(slt_conn)
    df.columns = get_col_name(vtc, df)
    return df



def get_col_name(vtc, pd_df) -> list:
    """
    读取文件名
    """
    n = pd_df.shape[1]
    return [vtc.description[i].name for i in range(n)]


def sql2excel(vtc, sql_out,  fil_name):
    stl_cnn = vtc.execute(sql_out).fetchall()
    df = pd.DataFrame(stl_cnn)
    df.columns = get_col_name(vtc, df)
    df.to_excel(fil_name, header = 'infer', index = False, encoding = 'gbk')
    print('Finished')



class get_sql_list():
    def __init__(self, sql_in =''):
        """
        param: sql_in 需要带有format  
            地市  用 lc  
            月份  用 mon  
        例子：
            sql_t = "select * from  zjbic.ofr_main_asset_mon_{lc} where bil_month = '{mon}'"
            gsql = get_sql_list(sql_t)
            sql_lst = gsql.sql_deal(mon=[201901, 201902], loc='all')
        """
        self.sql_in = sql_in

    def sql_array_lc(self, loc = 'all'):
        """
        对SQL填充地市  
        param loc： str  all / A-K  
        return: list
        """
        sql_lst = []
        lc_t = []
        if loc == 'all':
            lc_t = local_lst
        else:
            lc_t.extend(loc)
        for lc in lc_t:
            sql_lst.append(self.sql_in.format(lc=lc))
            print(f'Finished add {lc}')
        return sql_lst


    def sql_array_mon(self, mon: list, sql_l: list) -> list:
        """
        对一序列sql填充月
        param mon: list 
        param sql_l: list
        """
        sql_lst = []
        # 遍历月份
        for m in mon:
            # 遍历sql
            for sql_i in sql_l:
                try:
                    sql_lst.append(sql_i.format(mon = m))
                except:
                    print('该SQL无需填充月份， 或是月份format形式不对，需要改成 mon')
                    break
            print(f'finish add {m}')
        return sql_lst

    def sql_deal(self, mon=[], loc=[]) -> list:
        """
        将加月份加地市情况汇总
        """
        sql_lst = []
        if (mon == []) & (loc != []):
            sql_lst = self.sql_array_lc(loc=loc)
        elif (mon == []) & (loc == []):
            return self.sql_in
        elif (mon != []) & (loc == []):
            sql_lst = self.sql_array_mon(mon=mon, sql_l=[self.sql_in])
        else:
            lc_t = []
            if loc == 'all':
                lc_t = local_lst
            else:
                lc_t.extend(loc)
            for m in mon:
                for lc in lc_t:
                    sql_lst.append(self.sql_in.format(lc=lc, mon=m))
                print(f'Finished {m}')
        return sql_lst


if __name__ == '__main__':
    sql_t = "select * from  zjbic.ofr_main_asset_mon_{lc} where bil_month = '{mon}'"
    gsql = get_sql_list(sql_t)

    sql_lst = gsql.sql_deal(mon=[201901], loc='all')
    for i in sql_lst:
        print(i)

    sql_lst = gsql.sql_deal(mon=[201901, 201909], loc=['A', 'E'])
    for i in sql_lst:
        print(i)
