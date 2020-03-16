# python 3.6
# Author: Scc_hy
# Create date: 2020-01-06
# Function: 数据预处理

import datetime
import time
import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
from joblib import Parallel, delayed


# @staticmethod
def group_feature(df: pd.DataFrame, key: str, target: pd.DataFrame, aggs: list) -> pd.DataFrame:
    """
    聚合特征生成
    参考： https://github.com/jt120/tianchi_ship_2019/blob/master/working/201_train_1.ipynb
    param df: pd.DataFrame 处理的数据集  
    param key: str /list        需要groupby 的key  
    param target: str      聚合的特征  
    param agg: list        聚合的类型  
    return pd.DataFrame columns = [target, ag]
    例如：  
        aggs_list = ['max', 'min', 'mean', 'std', 'skew', 'sum']  
        group_feature(df, 'ship','x',aggs_list)  
    """
    agg_dict = {}
    for ag in aggs:
        agg_dict[f'{target}_{ag}'] = ag
    print(agg_dict)
    outdt = df.groupby(key)[target].agg(agg_dict).reset_index()
    return outdt

def cross_max_feat(dt,key):
    # 找两组特征中出现最多的特征及特征的出现次数
    # key = ['ship','x']
    target = key[-1]
    cnt_tmp = group_feature(dt, key, target, ['count'])
    cnt_tmp_max = group_feature(cnt_tmp, key[0], f'{target}_count', ['max'])
    cnt_tmp_max.columns = ['ship', f'{target}_count']
    cnt_tmp_max = pd.merge(cnt_tmp_max
                            , cnt_tmp.drop_duplicates(['ship', f'{target}_count'])
                            , on=['ship', f'{target}_count'], how='inner')
    return cnt_tmp_max



def union_dt(fil_path, out_path, tr_flg = True):
    """
    将文件合并
    """
    tr_te_f = 'train_dt.csv' if tr_flg else 'test_dt.csv'
    out_fil = os.path.join(out_path, tr_te_f)
    os.chdir(fil_path)
    fil_list = os.listdir()
    for fl_name in tqdm(fil_list):
        f = open(fl_name, encoding = 'utf-8')
        if tr_flg:
            f.seek(37)
        else:
            f.seek(32)
        f_cont = f.read()
        f_write = open(out_fil, 'a+', encoding = 'utf-8')
        f_write.write(f_cont)

        f_write.close()
        f.close()
    print('文件: {} 合并完毕.....'.format(tr_te_f))

"""
fl_name = r'E:\Competition\2020zhihui_haiyang\data\hy_round1_testA_20200102\7000.csv'
out_fil = r'E:\Competition\2020zhihui_haiyang\data\aaa.csv'
f = open(fl_name, encoding = 'utf-8')

f.seek(32)

f_cont = f.read()
f_write = open(out_fil, 'a+', encoding = 'utf-8')
f_write.write(f_cont)

f_write.close()
f.close()
print(f_cont)

with open(os.path.join(out_path, 'test_dt.csv'), encoding = 'utf-8') as f:
    f_Cont1 = f.readlines(1)
    f_Cont2 = f.readlines(2)
    f_Cont3 = f.readlines(3)
    print(f_Cont1)
    print(f_Cont2)
    print(f_Cont3)
"""

def zhhy_read_feat(path, test_mode=False) -> list:
    """
    智慧海洋数据预处理导入  
    """
    df = pd.read_csv(path)
    df = df.iloc[::-1]

    if test_mode:
        df_feat = [df['渔船ID'].iloc[0], df['type'].iloc[0]]
        df = df.drop(['type'], axis=1)
    else:
        df_feat = [df['渔船ID'].iloc[0]]

    df['time'] = df['time'].apply(
        lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
    df_diff = df.diff(1).iloc[1:]
    df_diff['time_seconds'] = df_diff['time'].dt.total_seconds()
    df_diff['dis'] = np.sqrt(df_diff['x']**2 + df_diff['y']**2)

    df_feat.append(df['time'].dt.day.nunique())
    df_feat.append(df['time'].dt.hour.min())
    df_feat.append(df['time'].dt.hour.max())
    df_feat.append(df['time'].dt.hour.value_counts().index[0])

    df_feat.append(df['速度'].min())
    df_feat.append(df['速度'].max())
    df_feat.append(df['速度'].mean())


    df_feat.append(df_diff['速度'].min())
    df_feat.append(df_diff['速度'].max())
    df_feat.append(df_diff['速度'].mean())
    df_feat.append((df_diff['速度'] > 0).mean())
    df_feat.append((df_diff['速度'] == 0).mean())

    df_feat.append(df_diff['方向'].min())
    df_feat.append(df_diff['方向'].max())
    df_feat.append(df_diff['方向'].mean())
    df_feat.append((df_diff['方向'] > 0).mean())
    df_feat.append((df_diff['方向'] == 0).mean())

    df_feat.append((df_diff['x'].abs() / df_diff['time_seconds']).min())
    df_feat.append((df_diff['x'].abs() / df_diff['time_seconds']).max())
    df_feat.append((df_diff['x'].abs() / df_diff['time_seconds']).mean())
    df_feat.append((df_diff['x'] > 0).mean())
    df_feat.append((df_diff['x'] == 0).mean())

    df_feat.append((df_diff['y'].abs() / df_diff['time_seconds']).min())
    df_feat.append((df_diff['y'].abs() / df_diff['time_seconds']).max())
    df_feat.append((df_diff['y'].abs() / df_diff['time_seconds']).mean())
    df_feat.append((df_diff['y'] > 0).mean())
    df_feat.append((df_diff['y'] == 0).mean())

    df_feat.append(df_diff['dis'].min())
    df_feat.append(df_diff['dis'].max())
    df_feat.append(df_diff['dis'].mean())

    df_feat.append((df_diff['dis']/df_diff['time_seconds']).min())
    df_feat.append((df_diff['dis']/df_diff['time_seconds']).max())
    df_feat.append((df_diff['dis']/df_diff['time_seconds']).mean())

    return df_feat


def zhhy_read_feat_new(path, test_mode=False) -> list:
    """
    智慧海洋数据预处理导入  
    """
    df = pd.read_csv(path)
    df = df.iloc[::-1]

    if test_mode:
        df_feat = [df['渔船ID'].iloc[0], df['type'].iloc[0]]
        df = df.drop(['type'], axis=1)
    else:
        df_feat = [df['渔船ID'].iloc[0]]

    df['time'] = df['time'].apply(
        lambda x: datetime.datetime.strptime(x, "%m%d %H:%M:%S"))
    df['hour'] = df['time'].dt.hour
    df_diff = df.diff(1).iloc[1:]
    df_diff['time_seconds'] = df_diff['time'].dt.total_seconds()
    df_diff['dis'] = np.sqrt(df_diff['x']**2 + df_diff['y']**2)

    # 将速度限制一下速度 
    df['hour_day'] = 0
    df[(df.hour > 8) & (df.hour < 18) , 'hour_day'] = 1
    ## 白天速度 
    df_tmp = df[(df['速度'] > 1) & (df['速度'] < 11) & (df.hour_day == 1), :]
    std_3n = np.std(df_tmp['速度'].value_counts().index[:3])
    std_al = np.std(df_tmp['速度'].value_counts().index[:])
    if std_3n < std_al:
        sp_nd = round(np.mean(df_tmp['速度'].value_counts().index[:3]), 3)
    else:
        sp_nd = round(np.mean(df_tmp['速度'].value_counts().index[:2]), 3)

    ## 晚上
    df_tmp_n = df[(df['速度'] > 1) & (df['速度'] < 11) & (df.hour_day == 0), :]
    std_3n = np.std(df_tmp_n['速度'].value_counts().index[:3])
    std_al = np.std(df_tmp_n['速度'].value_counts().index[:])
    if std_3n < std_al:
        sp_nd_n = round(np.mean(df_tmp_n['速度'].value_counts().index[:3]), 3)
    else:
        sp_nd_n = round(np.mean(df_tmp_n['速度'].value_counts().index[:2]), 3)


    df['x'] /= 50000.0                     # 获取经度值
    df['y'] /= 200000.0                     # 获取维度之维度值

    df_feat.extend([
        df['time'].dt.day.nunique(),
        # df['time'].dt.hour.min(),
        # df['time'].dt.hour.max(),
        df['time'].dt.hour.value_counts().index[0],

        df['速度'].min(),
        df['速度'].max(),
        df['速度'].mean(),

        df_diff['速度'].min(),
        df_diff['速度'].max(),
        df_diff['速度'].mean(),
        (df_diff['速度'] > 0).mean(),
        (df_diff['速度'] == 0).mean(), # 10

        df_diff['方向'].min(),
        df_diff['方向'].max(),
        df_diff['方向'].mean(),
        (df_diff['方向'] > 0).mean(),
        (df_diff['方向'] == 0).mean(),

        # (df_diff['x'].abs() / df_diff['time_seconds']).min(),
        (df_diff['x'].abs() / df_diff['time_seconds']).max(), # 16
        (df_diff['x'].abs() / df_diff['time_seconds']).mean(),
        (df_diff['x'] > 0).mean(),
        (df_diff['x'] == 0).mean(),

        # (df_diff['y'].abs() / df_diff['time_seconds']).min(),
        (df_diff['y'].abs() / df_diff['time_seconds']).max(),
        (df_diff['y'].abs() / df_diff['time_seconds']).mean(),
        (df_diff['y'] > 0).mean(),
        (df_diff['y'] == 0).mean(),

        df_diff['dis'].min(),
        df_diff['dis'].max(), # 25
        df_diff['dis'].mean(), # 26

        # (df_diff['dis']/df_diff['time_seconds']).min(),
        (df_diff['dis']/df_diff['time_seconds']).max(),
        (df_diff['dis']/df_diff['time_seconds']).mean(),
        ## 增加速度的前 3的均值  如果前三的标准差低于 总体的就取前三 否则取前2
        sp_nd,
        sp_nd_n
    ])

    return df_feat





def zhhy_read_feat_nd(path) -> list:
    """
    智慧海洋数据预处理导入  
    """
    df = pd.read_csv(path)
    df = df.iloc[::-1]
    try:
        df.columns = ['ship','x','y','v','d','time']
    except:
        df.columns = ['ship', 'x', 'y', 'v', 'd', 'time', 'type']
        df = df.drop(['type'], axis=1)
    df['x'] /= 50000.0                     # 获取经度值
    df['y'] /= 200000.0                     # 获取维度之维度值

    df_feat = [df['ship'].iloc[0]]

    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df_diff = df.diff(1).iloc[1:]
    df_diff['time_seconds'] = df_diff['time'].dt.total_seconds()
    df_diff['dis'] = np.sqrt(df_diff['x']**2 + df_diff['y']**2)

    # 将速度限制一下速度
    df['hour'] = df['time'].dt.hour
    df['hour_day'] = 0
    df.loc[(df.hour > 8) & (df.hour < 18), 'hour_day'] = 1
    ## 白天v
    try:
        df_tmp = df.loc[(df['v'] > 1) & (df['v'] < 11) & (df.hour_day == 1), :]
    except:
        df_tmp = df.loc[(df.hour_day == 1), :]
    std_3n = np.std(df_tmp['v'].value_counts().index[:3])
    std_al = np.std(df_tmp['v'].value_counts().index[:])
    if std_3n < std_al:
        sp_nd = round(np.mean(df_tmp['v'].value_counts().index[:3]), 3)
    else:
        sp_nd = round(np.mean(df_tmp['v'].value_counts().index[:2]), 3)

    ## 晚上
    try:
        df_tmp_n = df.loc[(df['v'] > 1) & (df['v'] < 11) & (df.hour_day == 0), :]
    except:
        df_tmp_n = df.loc[(df.hour_day == 0), :]

    std_3n = np.std(df_tmp_n['v'].value_counts().index[:3])
    std_al = np.std(df_tmp_n['v'].value_counts().index[:])
    if std_3n < std_al:
        sp_nd_n = round(np.mean(df_tmp_n['v'].value_counts().index[:3]), 3)
    else:
        sp_nd_n = round(np.mean(df_tmp_n['v'].value_counts().index[:2]), 3)


    df_tp = df.loc[(df['v'] > (sp_nd - std_3n) ) & (df['v'] < (sp_nd + std_3n) ), :].reset_index(drop = True)
    df_tp['time_hour'] = df_tp['time'].dt.hour
    
    ## 在一个小文件中进行特征衍生  
    # ## 增加作业区间的时间(最大-最小) 作业时间均值
    # self.tr_te['v_get_time'] = self.tr_te['ship'].map(train[['ship','v']].set_index('ship').to_dict()['v'])
    # tr_te_tmp = self.tr_te.loc[self.tr_te['v_get_time'] == self.tr_te['v'], ['ship', 'time']]\
    #                 .reset_index(drop =True)\
    #                 .copy(deep = True)
    # ## 1- 最多作业时间 作业速度出现最多的点 
    # tr_te_tmp['time'] = pd.to_datetime(tr_te_tmp['time'], format='%m%d %H:%M:%S')
    # tr_te_tmp['time_hour'] = tr_te_tmp['time'].dt.hour
    # t = cross_max_feat(tr_te_tmp, ['ship', 'time_hour'])

    # train = pd.merge(train, t, on='ship', how='left')
    # ## 2- 最多作业时间 的时间最大时间和最小时间
    # aggs_list_m = ['max', 'min']
    # t_max_min = group_feature(tr_te_tmp, 'ship', 'time', aggs_list_m)
    # t_max_min['diff'] = t_max_min['time_max'] - t_max_min['time_min']
    # t_max_min['diff_sc'] = t_max_min['diff'].dt.total_seconds()
    # t_max_min.drop('diff', axis = 1, inplace = True)
    # train = pd.merge(train, t_max_min, on='ship', how='left')



    df_feat.extend([
        df_diff['v'].mean(), # diff_v_mean

        (df_diff['x'].abs() / df_diff['time_seconds']).max(),  # diff_x_time_max
        (df_diff['x'].abs() / df_diff['time_seconds']).mean(), # diff_x_time_mean
        (df_diff['x'] > 0).mean(),  # diff_x_obe_mean
        (df_diff['x'] == 0).mean(), # diff_x_zero_mean

        (df_diff['y'].abs() / df_diff['time_seconds']).max(),  # diff_y_time_max
        (df_diff['y'].abs() / df_diff['time_seconds']).mean(), # diff_y_time_mean
        (df_diff['y'] > 0).mean(),  # diff_y_obe_mean
        (df_diff['y'] == 0).mean(), # diff_y_zero_mean

        df_diff['dis'].mean(),  # diff_dis_mean
        (df_diff['dis']/df_diff['time_seconds']).mean(), # diff_dis_time_mean
        sp_nd,  # sp_nd
        sp_nd_n
    ])

    return df_feat



class M_ship_dt():
    def __init__(self, all_tr, all_te,
                train_data_root, test_data_root):
        self.tr = all_tr
        self.te = all_te
        self.train_data_root = train_data_root
        self.test_data_root = test_data_root
    
    def get_first_tr(self):
        print('load train data.....')
        os.chdir(self.train_data_root)
        train_feat = Parallel(n_jobs=10)(delayed(zhhy_read_feat_nd)(path)
                                    for path in tqdm(os.listdir()) )
        train_feat = pd.DataFrame(train_feat)
        return train_feat

    def get_first_te(self):
        print('load test data.....')
        os.chdir(self.test_data_root)
        test_feat = Parallel(n_jobs=10)(delayed(zhhy_read_feat_nd)(path)
                                        for path in tqdm(os.listdir()) )
        test_feat = pd.DataFrame(test_feat)
        test_feat = test_feat.sort_values(by=0)

        return test_feat

    def get_first_feat_cont(self, tr, te):
        tr_te1 = pd.concat([tr, te], axis=0, ignore_index=True)
        tr_te1.columns = ['ship','diff_v_mean', 'diff_x_time_max', 'diff_x_time_mean', 'diff_x_obe_mean'
                    , 'diff_x_zero_mean', 'diff_y_time_max', 'diff_y_time_mean', 'diff_y_obe_mean'
                    , 'diff_y_zero_mean', 'diff_dis_mean', 'diff_dis_time_mean', 'sp_nd', 'sp_nd_n']
        tr_te1.reset_index(drop = True, inplace = True)
        return tr_te1



    def get_tr_te(self):
        tr_te = pd.concat([self.tr, self.te], axis=0, ignore_index=True)
        tr_te.reset_index(drop=True, inplace=True)
        tr_te['type'] = tr_te['type'].map({'围网': 0, '刺网': 1, '拖网': 2})
        tr_te['x'] /= 50000.0                     # 获取经度值
        tr_te['y'] /= 200000.0                     # 获取维度之维度值

        tr_te['time'] = pd.to_datetime(tr_te['time'], format='%m%d %H:%M:%S')
        # df['month'] = df['time'].dt.month
        # df['day'] = df['time'].dt.day
        tr_te['date'] = tr_te['time'].dt.date
        tr_te['hour'] = tr_te['time'].dt.hour
        # df = df.drop_duplicates(['ship','month'])
        tr_te['weekday'] = tr_te['time'].dt.weekday
        self.tr_te = tr_te

    def extract_feature(self, df, train):
        # 最大 最小 均值 标准差 偏度 总值
        aggs_list = ['max', 'min', 'mean', 'std', 'skew', 'sum']
        
        t = group_feature(df, 'ship', 'x', aggs_list)
        train = pd.merge(train, t, on='ship', how='left')
        t = group_feature(df, 'ship', 'x', ['count']) # 记录次数
        t.columns = ['ship', 'x_count_cnt']
        train = pd.merge(train, t, on='ship', how='left')

        t = group_feature(df, 'ship', 'y', aggs_list)
        train = pd.merge(train, t, on='ship', how='left')

        t = group_feature(df, 'ship', 'v', aggs_list)
        train = pd.merge(train, t, on='ship', how='left')
        t = group_feature(df, 'ship', 'd', aggs_list)
        train = pd.merge(train, t, on='ship', how='left')


        train['x_max_x_min'] = train['x_max'] - train['x_min']
        train['y_max_y_min'] = train['y_max'] - train['y_min']
        train['y_max_x_min'] = train['y_max'] - train['x_min']
        train['x_max_y_min'] = train['x_max'] - train['y_min']
        # 斜率
        train['slope'] = train['y_max_y_min'] / \
            np.where(train['x_max_x_min'] == 0, 0.001, train['x_max_x_min'])
        train['area'] = train['x_max_x_min'] * train['y_max_y_min']

        mode_hour = df.groupby('ship')['hour'].agg(
            lambda x: x.value_counts().index[0]).to_dict()
        train['mode_hour'] = train['ship'].map(mode_hour)

        t = group_feature(df, 'ship', 'hour', ['max', 'min'])
        train = pd.merge(train, t, on='ship', how='left')

        hour_nunique = df.groupby('ship')['hour'].nunique().to_dict()
        date_nunique = df.groupby('ship')['date'].nunique().to_dict()
        train['hour_nunique'] = train['ship'].map(hour_nunique)
        train['date_nunique'] = train['ship'].map(date_nunique)

        t = df.groupby('ship')['time'].agg(
            {'diff_time': lambda x: np.max(x)-np.min(x)}).reset_index()
        t['diff_day'] = t['diff_time'].dt.days
        t['diff_second'] = t['diff_time'].dt.seconds
        train = pd.merge(train, t, on='ship', how='left')
        print('OK')

        return train
    
    def get_train(self):
        train = self.tr_te.drop_duplicates('ship')[['ship','type']].reset_index()
        t_list = ['x', 'y', 'd', 'v']
        for target in t_list: # 出现时间最多的特征
            t = cross_max_feat(self.tr_te, ['ship',target])
            train = pd.merge(train, t, on='ship', how='left')
        
        # ## 先在总体进行特征衍生 测试一 20190113 11：00   f1 = 0.875
        # ## 增加作业区间的时间(最大-最小) 作业时间均值
        # self.tr_te['v_get_time'] = self.tr_te['ship'].map(train[['ship','v']].set_index('ship').to_dict()['v'])
        # tr_te_tmp = self.tr_te.loc[self.tr_te['v_get_time'] == self.tr_te['v'], ['ship', 'time']]\
        #                 .reset_index(drop =True)\
        #                 .copy(deep = True)
        # ## 1- 最多作业时间 作业速度出现最多的点 
        # tr_te_tmp['time'] = pd.to_datetime(tr_te_tmp['time'], format='%m%d %H:%M:%S')
        # tr_te_tmp['time_hour'] = tr_te_tmp['time'].dt.hour
        # t = cross_max_feat(tr_te_tmp, ['ship', 'time_hour'])

        # train = pd.merge(train, t, on='ship', how='left')
        # ## 2- 最多作业时间 的时间最大时间和最小时间
        # aggs_list_m = ['max', 'min']
        # t_max_min = group_feature(tr_te_tmp, 'ship', 'time', aggs_list_m)
        # t_max_min['diff'] = t_max_min['time_max'] - t_max_min['time_min']
        # t_max_min['diff_sc'] = t_max_min['diff'].dt.total_seconds()
        # t_max_min.drop('diff', axis = 1, inplace = True)
        # train = pd.merge(train, t_max_min, on='ship', how='left')

        return train

    def get_dt_quick(self):
        """
        合并所有操作, 最终输出建模用dt
        """
        # te 读取可能会出问题
        tr = self.get_first_tr()
        te = self.get_first_te()
        t = self.get_first_feat_cont(tr, te)

        self.get_tr_te()
        train = self.get_train()
        train = pd.merge(train, t, on='ship', how='left')
 
        out_dt = self.extract_feature(self.tr_te, train)
        return out_dt
