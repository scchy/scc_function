# python 3.6
# author: Scc_hy 
# create date: 2019-12-24
# Function： 数据探索函数

import pandas as pd
import random
import copy
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns 
import matplotlib.pyplot as plt 
from  matplotlib import gridspec 
from scc_function import data2excel ,progressing
import sys

large = 22; mid = 16; small = 12
param ={'axes.titlesize': large,
        'legend.fontsize': mid,
        'figure.figsize': (16, 10),
        'axes.labelsize': mid,
        'axes.titlesize': mid,
        'xtick.labelsize': mid,
        'ytick.labelsize': mid,
        'figure.titlesize': large}
plt.rcParams.update(param)
plt.style.use('ggplot')
# sns.set_style('white')
import missingno as msn
from tqdm import tqdm

def M_reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    压缩大的df
    https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def M_data_scan(df: pd.DataFrame) -> None:
    """
    数据整体描述  
        维度、占内存大小、字段类型、缺失
    """
    print('# -------------------------')
    print('#        维度与数据大小     ')
    print('# -------------------------')
    print("数据 {} 行 {} 列".format(df.shape[0], df.shape[1]))
    print("数据占内存：{:.2f}MB".format(df.memory_usage().sum() / 1024**2))
    print("数据集的特征类型：\n", df.dtypes)
    print('# -------------------------')
    print('#   特征类型和缺失情况 ')
    print('# -------------------------')
    cnt = 0 
    for col in df.columns:
        dtp =  df[col].dtypes
        mis_cnt = df[col].isna().sum()
        if mis_cnt > 0:
            print("特征名称:{}, 特征类型：{}, 缺失数量:{}".format(col, dtp, mis_cnt))
            cnt += 1 
    msg = '所有变量均无缺失' if cnt == 0 else '总共 %d 个缺失' % (cnt)
    print(msg)
    msn.bar(df)
    plt.show()




class explore_2_excel(object):
    """
    数据的文本和数值型特征
    """
    def __init__(self, out_file):
        self.out_file = out_file
        self.sheet_name_pre = ''
    
    def load_df(self, df: pd.DataFrame):
        self.df = df
        self.obj_col = list(df.dtypes.index[df.dtypes == 'object'])
        self.num_col = list(df.dtypes.index[df.dtypes != 'object'])
        self.ID_judge()

    def ID_judge(self) -> list:
        """
        判断是否是ID
        """
        self.id_list = []
        m = self.df.shape[0] / 2
        for i in self.obj_col:
            if 'id' in i.lower():
                self.id_list.append(i)
            len_n = len(self.df[i].unique())
            if len_n > m:
                self.id_list.append(i)


    def Object_cat2csv(self, sheet_name = 'obj_cat'):
        """
        输出到excel 观察变量
        记录将文本转变的因子\n
        Author: Scc_hy\n
        """
        sheet_name = self.sheet_name_pre + sheet_name
        m, n = self.df.shape
        out_dt = pd.DataFrame(np.array(['a', 'tx', 999999.00]).reshape(1,3), columns= ['col_name', 'col_cat', 'cnt'])
        out_col = ['col_name', 'col_cat', 'cnt']
        out_dt['cnt'] = out_dt['cnt'].astype(np.float)
        view_col = [i for i in self.obj_col if i not in self.id_list]
        if len(view_col) == 0:
            return print('除{}，无其他文本字段'.format(self.id_list))
        else:
            p = progressing.My_Progress(view_col, width = 45)
            for i in view_col:
                msg = p.progress()
                # print('目前处理(特征): {}, 进度: {}'.format(i, msg))
                print('目前处理(文本特征): {0:{2}<10}, 进度: {1}'.format(i, msg, chr(12288)))
                dt_i = pd.DataFrame(data = self.df[i].value_counts().reset_index())
                dt_i.columns = ['col_cat', 'cnt']
                dt_i['col_name'] = i
                out_dt = pd.concat([out_dt, dt_i[out_col]], axis = 0, ignore_index = True)

            out_dt['cnt_rate'] = out_dt['cnt']  / m
            out_dt.drop(0, axis = 0, inplace = True)
            data2excel.data2excel(self.out_file , out_dt, sheet_name)
            return print('已经将文本字段的字段内容占比输出到:\n \t{}--{}\n'.format(self.out_file, sheet_name))

    def num_cat2csv(self, sheet_name = 'num_cat', scanmethod = '5sd'):
        """搜索极端值并进行简单报告  
        scanmethod: 极端值确认标准， 5sd 超过均数+-5被标准差  
        返回: 包括全部数值变量搜索结果的数据框  
        """
        sheet_name = self.sheet_name_pre + sheet_name
        dfres = pd.DataFrame(columns = ['记录数','均值', '标准差', '最小值', '最大值', '小值超界', '大值超界'])
        p = progressing.My_Progress(self.num_col, width = 45)           
        for col in self.num_col:
            msg = p.progress()
            # print('目前处理: {}, 进度: {}'.format(col, msg))
            print('目前处理(数值特征): {0:{2}<10}, 进度: {1}'.format(col, msg, chr(12288)))
            try:
                if scanmethod == '5sd':
                    cnt = len(self.df[col])
                    v_std = self.df[col].std()
                    v_mean = self.df[col].mean()
                    dfres.loc[col, :] = [cnt, v_mean, v_std,
                                    self.df[col].min(), self.df[col].max(), 
                                    sum(self.df[col] < v_mean - 5 * v_std), 
                                    sum(self.df[col] > v_mean + 5 * v_std)
                                    ]
            except: # 非数值变量
                print('{}无法分析'.format(col))
        dfres.reset_index(inplace = True)
        data2excel.data2excel(self.out_file , dfres, sheet_name)
        return print('已经将数值字段的字段内容占比输出到:\n \t{}---{}\n'.format(self.out_file, sheet_name))
    
    def cat_exp2excel(self, sheet_name_pre = ''):
        self.sheet_name_pre = sheet_name_pre
        self.Object_cat2csv()
        self.num_cat2csv()



## 单变量分析
## 生成分组条图/百分条图
def M_GphDes(x, y, plt_type = '%', printdata = True):
    """ 
    param x: 自变量  
    param y: 因变量  
    param plt_type: %要求绘制百分比条图  
    param printdata: bool 是否列出原始数据  
    例子：
        M_GphDes(vname_bin, dfana[yvar], plt_type = '%')
        plt.show()
    """
    if plt_type == '%':
          tmp_df = pd.crosstab(index = y, columns = x, normalize = 'columns')
    else:
          tmp_df = pd.crosstab(index = y, columns = x)

    if printdata:
        print(tmp_df.T)

    base = pd.DataFrame(columns = tmp_df.columns)
    colorstep0 = 1 / len(tmp_df.index)
    for i in range(len(tmp_df.index)):
        if i == 0:
            base.loc[0,:] =  tmp_df.sum().values
            colorstep = colorstep0 / 2 # 避免最终出现纯白直条 
            sns.barplot(data = base,
                        color = str(colorstep), 
                        label = tmp_df.index[i])
            
        else:
            base.loc[0,:] -= tmp_df.loc[tmp_df.index[i - 1]]
            sns.barplot(data = base,
                    color = str(colorstep), 
                    label = tmp_df.index[i])
        colorstep += colorstep0       
    plt.title( "Percent Graph" if plt_type == '%' else 'Frequency Graph')
    plt.xticks(rotation = 55) 
    plt.legend()


def M_DesC(dfname, xvar, yvar):
    """
    对分类变量的预分析函数  
    图形可做成子图形式一起呈现，但代码更加复杂  
    例：
        M_DesC(dfana, 'xvar', 'yvar')
    """
    print(dfname[xvar].value_counts())
    plt.figure(figsize = (16, 8))
    plt.subplot(131)
    dfname[xvar].value_counts().plot.bar(color = '#666666', sort_columns = True)
    plt.title('Feature Counts Graph')
    plt.xticks(rotation=55)
    print('变量名称: ', xvar)
    plt.subplot(132)
    M_GphDes(dfname[xvar], dfname[yvar], plt_type = '')
    plt.subplot(133)
    M_GphDes(dfname[xvar], dfname[yvar], plt_type = '%')
    plt.show()



# -------------------------
#  判断特征是否具有单调性
## 用线性回归的p
# -------------------------
from statsmodels.formula.api import ols
def incresing(v: pd.Series) -> bool:
    """
    判断特征是否单调
    """
    y = np.arange(len(v))
    x = np.array(v)
    data = pd.DataFrame({'x':x, 'y':y})
    model = ols('y~x', data).fit()
    return 0 if model.f_pvalue > 0.001 else 1

# fea_cols = [i for i in dt_train.columns if 'var' in i]
# for col in tqdm(fea_cols):
#     incre_bool = incresing(dt_train[col])
#     if incre_bool == 1 :
#         print(col)




import lightgbm as lgb
import time
from sklearn.model_selection import KFold,StratifiedKFold
# 已经合并到 model_funcs
def validation_prediction_lgb(X,y,feature_names, ratio =1, X_test = None,istest = False):
    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    params = {
    'bagging_freq': 5, 
    'boost_from_average':'false',
    'boost': 'gbdt',
    'learning_rate': 0.01,
    'max_depth': 5,
    'metric':'auc',
    'min_data_in_leaf': 50,
    'min_sum_hessian_in_leaf': 10.0,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1}
    importances = pd.DataFrame() 
    if istest:
        prediction = np.zeros(len(X_test))
    models = []
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        weights = [ratio  if val == 1 else 1 for val in y_train]
        
        train_data = lgb.Dataset(X_train, label=y_train,  weight=weights)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        model = lgb.train(params,train_data,num_boost_round=20000,
                        valid_sets = [train_data, valid_data],verbose_eval=200,early_stopping_rounds = 200)
        
        imp_df = pd.DataFrame() 
        imp_df['feature']  = feature_names
        imp_df['split']    = model.feature_importance()
        imp_df['gain']     = model.feature_importance(importance_type='gain')
        imp_df['fold']     = fold_n + 1
        
        importances = pd.concat([importances, imp_df], axis=0)
        
        models.append(model)
        if istest == True:
            prediction += model.predict(X_test, num_iteration=model.best_iteration)/5
    if istest == True:     
        return models,importances, prediction
    else:
        return models,importances


# ===================================================================
# 智慧海洋
# ===================================================================
def show_path(train, type_name):
    """
    随机抽取画图
    """
    ids = train[train['type']==type_name]['ship'].unique()
    ids = [ids[np.random.randint(len(ids))] for x in range(10)]
    t = train[train['ship'].isin(ids)]

    f, ax = plt.subplots(5,2, figsize=(8,20))
    for index, cur_id in enumerate(ids):
        cur = t[t['ship']==cur_id]
        i = index//2
        j = index % 2
        ax[i,j].plot(cur['x'], cur['y'])
#         if i==0 and j==0:
        ax[i,j].set_title(cur_id)
    plt.show()


