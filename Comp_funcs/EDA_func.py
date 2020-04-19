# python 3.6
# author: Scc_hy 
# create date: 2019-12-24
# Function： 数据探索函数&数据清洗

import random
import copy
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
import seaborn as sns 
import matplotlib.pyplot as plt 
from  matplotlib import gridspec 
from scc_function import data2excel ,progressing
import numpy as np
import pandas as pd 
from datetime import datetime
from tqdm import tqdm
import sys, os
import warnings 
warnings.filterwarnings(action = 'ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from pyecharts.charts import Bar, Pie ,Page, Radar
import pyecharts.options as opts 
import scipy.stats as st
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

# ================================================================
#                          零、  实用探索辅助函数
# ================================================================
def get_now():
    """
    查看现在的时间
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def group_feature(df: pd.DataFrame, key: str, target: pd.DataFrame, aggs: list) -> pd.DataFrame:
    """
    聚合特征生成
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


def reduce_df_mem(df: pd.DataFrame) -> pd.DataFrame:
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


# ================================================================
#                          一、  数据探索
# Explore_func()   --->>> 基本是查看为主，查看缺失，分布，异常  
# EDA_func()   --->>> 着重于发现特点, 查看分布, 目标与变量之间的关系(分类分布， 连续相关性) 
# ================================================================
## 1-1 基本查看
class Explore_func():
    """
    基本是查看为主，查看缺失，分布，异常  
    :  get_losedf(df, plt_flg=False) --> 查看缺失  
    :  Sta_inf(dt)                  --> 查看pd.DataFrame分布或某个特征分布  
    :  tr_te_cols_distribute        --> 查看测试集和训练集的分布差异    
    :  outliers_proc                --> 处理异常值，考虑是否删除 （依据test_flg & delete_flg)     
    :  num_target_distrbuted(df_all, targert, plt_flg = True) --> 查看数值型特征的分布  
    :  cat_target_distrbuted(df_all, targert, plt_flg = True) --> 查看分类型特征的分布  
    :  compute_woe1(self, x:pd.Series, y:pd.Series, na = -1) --> 计算两个类别Series (分箱-目标二分类) 的woe 和iv
    """
    def get_losedf(self, df_all, plt_flg = False):
        """
        查看缺失
        """
        ls_df = pd.DataFrame(df_all.isna().sum()).reset_index()
        ls_df.columns = ['feat', 'ls_cnt']
        ls_df['ls_rate'] = ls_df.ls_cnt / df_all.shape[0] * 100 
        ls_df = ls_df.loc[ls_df.ls_cnt != 0 ,:].sort_values(by = 'ls_cnt', ascending = False).reset_index(drop = True)
        if plt_flg:
            bar_p = (
                        Bar()
                        .add_xaxis(ls_df.feat.tolist())
                        .add_yaxis('流失特征柱状图',ls_df.ls_cnt.tolist())
                    )
            return ls_df, bar_p

        return ls_df 


    def Sta_inf(self, dt):
        """
        查看数据集或者一个col的分布
        [np.min, np.max, np.mean, np.ptp, np.std, np.var]
        """
        print('_min:', np.min(dt))
        print('_max:', np.max(dt))
        print('_mean:', np.mean(dt))
        print('_ptp:', np.ptp(dt))
        print('_std:', np.std(dt))
        print('_var:', np.var(dt))
        if dt.shape[1] == 1:
            print('_skew:', np.skew(dt))
            print('_kurt:', np.kurt(dt))


    def tr_te_cols_distribute(self, df_all, cols_lst, test_flg, func = 'skew'):
        """
        查看测试集和训练集的分布差异  
        注意用 skew kurt的时候需要时  
        param df_all:     pd.DataFrame  
        param  test_flg:  test_flg 测试集标识 np.array bool  
        param cols_lst:   list 查看特征分布的字段  
        param func: str: skew, nunique, kurt  
         
        df_func出来后，处理方式建议: 

        """
        df_func = pd.DataFrame(columns = [f'tr_{func}', f'te_{func}'])
        f = eval(f'pd.Series.{func}')
        for col_i in tqdm(cols_lst):
            tr_i = f(df_all.loc[~test_flg, col_i])
            te_i = f(df_all.loc[test_flg, col_i])
            df_func.loc[col_i, f'tr_{func}'] = tr_i
            df_func.loc[col_i, f'te_{func}'] = te_i

        plt.figure(figsize=(25, 8))
        plt.title(f'The {func} distribution of all feats')
        plt.plot(df_func.index.tolist(), df_func[f'tr_{func}'].tolist(), label = 'tr')
        plt.plot(df_func.index.tolist(), df_func[f'te_{func}'].tolist(), label = 'te')
        plt.xticks(rotation = 75)
        plt.legend()
        plt.show()
        return df_func


    def box_plot_outliers(self, data_ser, bax_scale = 3):
        """
        利用箱线图去除异常值:
        param  data_ser: pd.Series
        param  bax_scale: 
        """
        iqr = bax_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        low_bool = data_ser < val_low
        up_bool = data_ser >  val_up
        return (low_bool, up_bool), (val_low, val_up)


    def outliers_proc(self, df, col, test_flg, delete_flg = False, scale = 3):
        """
        处理异常值，考虑是否删除 （依据test_flg & delete_flg）  
        仅仅对一个特征进行操作  
        : param df: pd.DataFrame  
        : param col: str  
        : param test_flg: np.array(bool)  
        : param delete_flg: bool 是否删除异常点  
        : param scale:  int  异常判断标准范围 
        """
        dt_tmp = df.copy(deep=True)
        dt_ser = dt_tmp[col]
        bool_, value_ =  box_plot_outliers(dt_ser, bax_scale = scale)
        # 人工删除 并判断是否要删除 
        low_test_len = sum((bool_[0])| test_flg)
        uper_test_len = sum((bool_[1])| test_flg)
        delete_flg_low = False if low_test_len > 0 else delete_flg
        delete_flg_up = False if uper_test_len > 0 else delete_flg
        print(f'Allowed Delete low is: <{delete_flg_low}> ,Allowed Delete upper is: <{delete_flg_up}>')
        if delete_flg_low:
            del_index_low = df.loc[ bool_[0], col].index.tolist()
            print(f'Delete lower number is: {len(del_index_low)}')
            dt_tmp = dt_tmp.drop(del_index_low)
            dt_tmp.reset_index(drop = True, inplace = True)
            print(f'Now the shape of data is {dt_tmp.shape}')
        if delete_flg_up:
            del_index_up = df.loc[ bool_[1], col].index.tolist()
            print(f'Delete upper number is: {len(del_index_up)}')
            dt_tmp = dt_tmp.drop(del_index_up)
            dt_tmp.reset_index(drop = True, inplace = True)
            print(f'Now the shape of data is {dt_tmp.shape}')
            
        print(f'Description of data less than the lower bound  is:')
        print(df.loc[df[bool_[0]].index, col].describe())
        print(f'Description of data larger than the upper bound is:')
        print(df.loc[df[bool_[1]].index, col].describe())
        
        
        if delete_flg:
            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize =(10, 7))
            sns.boxplot(y=col, data = df, palette='Set1', ax=ax[0])
            sns.boxplot(y=col, data = dt_tmp, palette='Set1', ax=ax[1])
            plt.show()
            return dt_tmp
        
        plt.figure(figsize =(10, 7))
        sns.boxplot(y=col, data = df, palette='Set1')
        plt.show() 
        return dt_tmp


    def compute_woe(self, x:pd.Series, y:pd.Series, na = -1) -> set:
        """
        计算两个类别Series (分箱-目标二分类) 的woe 和iv
        param  x:pd.Series  分箱特征
        param  y:pd.Series 目标二分类
        param  na:  缺失填补策略 -1
        return： set(pd.DataFrame, np.float)
        """
        try:
            tb = pd.crosstab(x.fillna(na),y,dropna=False,margins=True)
        except:
            tb = pd.crosstab(x,y,dropna=False,margins=True)
        # 读取两个类的总数
        pos = pd.value_counts(y)[1]
        neg = pd.value_counts(y)[0]

        tb['rat'] = tb.iloc[:,1] / tb.loc[:, 'All']
        tb.fillna(0, inplace=True)
        bad_rat = tb.iloc[:, 0] / neg
        good_rat = tb.iloc[:,1] / pos
        # 对于空值和无限大处理
        good_bad_rate = (good_rat / bad_rat).fillna(0).replace(np.inf, 2**31)
        # 处理woe的正无穷和负无穷
        tb['woe'] = np.log(good_bad_rate) * 100
        tb['woe'] = tb['woe'].replace(np.inf, np.log(2**31)).replace(-np.inf, 0)
        bad_dev = good_rat - bad_rat
        iv = sum(bad_dev * tb['woe']) /100
        return tb, iv

        # 异常处理
        def deloutliers(self, pd_series, l = 0.01, u = 0.99):
            """
            异常处理， 盖帽法
            """
            a=pd_series.quantile(l)
            b=pd_series.quantile(u)
            pd_series=pd_series.map(lambda x:b if x>b else a if x < a else x)
            return pd_series



class EDA_func():
    """
    着重于发现特点, 查看分布, 目标与变量之间的关系(分类分布， 连续相关性)  
    :  get_trte_pie(df_all, target) --> 查看一个训练集和预测集比例  
    :  single_numfeat_distrbuted(df_all, targert, plt_flg = True) --> 查看一个数值型特征的分布拟合  
    :  multi_catfeat_distrbuted(df_all, targert, plt_flg = True) --> 查看分类特征的分布    
    :  get_objct_label_draw -> 绘制分类特征和label之间的分布关系  
         子函数: 
            get_label_x_group -> 计算特征的和label的聚合信息  
            bar_stack1(dt, feat, target = 'label',type_draw = '%') -> 绘制一个特征的百分比堆叠图
    """
    def single_numfeat_distrbuted(self, df_all:pd.DataFrame, targert:str, plt_flg = True) -> pd.DataFrame:
        """
        一般查看回归的目标变量  
        查看一个数值型特征的分布拟合  
        : param df_all:  pd.DataFrame  
        : param targert: str  
        : param plt_flg: bool  
        """
        y = df_all[targert].tolist()
        plt.figure(figsize=(16,8))
        plt.subplot(131)
        sns.distplot(y, kde=False, fit = st.johnsonsu)
        plt.subplot(132)
        sns.distplot(y, kde=False, fit = st.norm)
        plt.subplot(133)
        sns.distplot(y, kde=False, fit = st.lognorm)
        plt.show() 
        skew_, kurt_= df_all[targert].skew(), df_all[targert].kurt()
        msg = f'Skewness: {skew_}, Kurtosis: {kurt_}'
        print(msg)
        return skew_, kurt_

    def multi_catfeat_distrbuted(self, tr_dt:pd.DataFrame, col_obj:list, plt_flg = True) -> pd.DataFrame:
        """
        查看分类特征的分布(pyecharts)  
        : param tr_dt:  pd.DataFrame  
        : param col_obj: list  
        : param plt_flg: bool  
        """
        aggs_list = ['count']
        Page_col = Page("类型特征特征分布情况",layout=Page.SimplePageLayout)
        df_out = pd.DataFrame()
        for i in col_obj:
            df_tmp = group_feature(tr_dt, i, i, aggs_list)
            if df_tmp.shape[0] < 1:
                pass
            else:
                df_tmp.columns = ['feat_', 'cnt']
                df_tmp.loc[:,'col_name'] = i 
                df_out = pd.concat([df_out, df_tmp],axis = 0 ,ignore_index =True)
                tmp_pie = (
                    Pie().add(f'特征: {i} 的种类分布', [list(z) for z in zip(df_tmp.iloc[:,0],df_tmp.iloc[:,1])]
                        ,radius=["30%", "75%"]
                        ,rosetype = "radius" if df_tmp.shape[0] >= 3 else None)
                        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{c}:({d}%)"))
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title = f'特征: {i} 的种类分布')
                            ,legend_opts=opts.LegendOpts(
                            type_="scroll", pos_top="20%", pos_left="80%", orient="vertical"
                        ))
                )
                Page_col.add(tmp_pie)
        return Page_col, df_out

    def get_trte_pie(self, df_all, target):
        """
        查看一个训练集和预测集比例
        : param df_all: pd.DataFrame  
        : param target: str 目标特征
        """
        tr_te_cnt = list(df_all[target].isna().value_counts())
        tr_te_pie = (
            Pie().add('', [list(z) for z in zip(['训练', '测试'],tr_te_cnt)]
                    ,radius=["30%", "75%"]
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}:{c}:({d}%)"))
            .set_global_opts(
                title_opts=opts.TitleOpts(title = '训练集和测试集占比')
            )
        )
    return tr_te_pie


    def get_label_x_group(dt, feat, target = 'label'):
        """
        计算特征的和label的聚合信息  
        param df: 数据集  
        param feat: 特征  
        param target：目标变量  
        """
        t = dt.groupby([target,feat])[target].agg({'cnt':'count'}).reset_index()
        t['g_sum'] = t.groupby(target)['cnt'].transform('sum')
        t['g_cnt_p'] =  t['cnt']/t['g_sum']
        return t


    def bar_stack1(dt, feat, target = 'label',type_draw = '%'):
        """
        绘制一个特征的百分比堆叠图  
        param dt: 数据集  
        param target: string  
        param type_draw: 绘制类型 频率, 百分比堆叠图 ['%', 'cnt'] 
        """
        t = get_label_x_group(dt, feat, target )
        target_col = dt[target].unique().tolist()
        draw_dict = {'%': 'g_cnt_p', 'cnt':'cnt', 'percent':'g_cnt_p'}
        try:
            draw_v = draw_dict[type_draw]
        except:
            raise KeyError("print right type_draw in ['%', 'percent', 'cnt']")
        c = (
            Bar()
            .add_xaxis(target_col)
        )
        val_arr,feat_lst = [], []
        for feat_i in t[feat].unique():
            val_lst = []
            for label_i in target_col:
                try:
                    val_lst.append(
                        t.loc[(t.loc[:,target] == label_i) & (t.loc[:,feat] == feat_i)
                            , draw_v].values[0]
                    )
                except:
                    val_lst.append(0.00)
            c.add_yaxis(f"{feat_i}", val_lst, stack="stack1")

        c.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        msg = '\nPercent Graph' if type_draw == '%' else '\nFrequence Graph'
        c.set_global_opts(title_opts=opts.TitleOpts(title=f"{feat}{msg}"))
        return c


    def get_objct_label_draw(dt:pd.DataFrame, col_list:list) -> Page:
        """
        绘制分类特征和label之间的分布关系  
        """
        page_bar = Page("类型特征特征分布情况",layout=Page.SimplePageLayout)
        for feat_i in col_list:
            c = bar_stack1(dt, feat_i, target='label' ,type_draw = '%')
            page_bar.add(c)
    #         c = bar_stack1(dt, feat_i, type_draw = 'cnt')
    #         page_bar.add(c)
            print(f'Finished Draw {feat_i}')
        return page_bar


class cat_target_Radar():
        def max_min_df(self, df, cols, target):
            """
            maxmin标准化
            """
            df_tmp = df[cols+[target]].copy(deep=True)
            df_tmp = df_tmp.fillna(0)
            df_tmp[cols] = df_tmp[cols].apply(lambda col: (col - col.min()) / (col.max() - col.min()))
            return df_tmp

        def get_view_dt(self, df_all, target, view_label, call_cols, dct=None):
            """
            获取目标变量两类特征（view_label）对应的call_cols 的均值，并求出差值
            param df_all: 数据集 pd.DataFrame
            param target: object 目标变量
            param view_label: list 要观察的目标变量的两个类型
            param call_cols: list 要进行比对的特征
            param dct: dict 是否用字典将特征转换为中文
            """
            df_tmp = self.max_min_df(df_all.loc[df_all[target].isin(view_label), :], call_cols, target)
            df_view = df_tmp.groupby(by = target)[call_cols].mean().T
            df_view['diff'] = (df_view[view_label[0]]  - df_view[view_label[1]]).map(abs)
            df_view = df_view.sort_values(by='diff',ascending=False).reset_index()
            if dct is None:
                pass
            else:
                df_view['index'] = df_view['index'].map(dct)
            return df_view

        def view_radar(self, df_view):
            """
            画雷达图
            
            例子：
                page_ = Page()
                for v in view_lst:
                    view_label[1] = v
                    df_view = get_view_dt(df_all, 'label', view_label, call_cols, call_dct)
                    c = view_radar(df_view)
                    page_.add(c)
                    print(f'Finished {v}')
                print('Finished all')  
                page_.render_notebook()
            """
            c_schema = []
            try:
                add_name1, add_name2 = df_view.columns[1], df_view.columns[2]
                add_value1, add_value2 = [df_view.iloc[list(range(12)), 1].tolist()], [df_view.iloc[list(range(12)), 2].tolist()]
                max_v = max(max(add_value1[0]), max(add_value2[0])) 
                min_v = min(min(add_value1[0]), min(add_value2[0])) 
                max_v_set = 1 if max_v > 0.65  else max_v + min_v 
                for i in df_view.loc[list(range(12)), 'index']:
                    c_schema.append({"name": i, "max": max_v_set, "min": 0})      
            except:
                add_name1, add_name2 = df_view.columns[1], df_view.columns[2]
                add_value1, add_value2 = [df_view.iloc[:, 1].tolist()], [df_view.iloc[:, 2].tolist()]
                max_v = max(max(add_value1[0]), max(add_value2[0])) 
                min_v = min(min(add_value1[0]), min(add_value2[0])) 
                max_v_set = 1 if max_v > 0.65  else max_v + min_v 
                for i in df_view.loc[:, 'index']:
                    c_schema.append({"name": i, "max": max_v_set, "min": 0})      

            c = (
                Radar()
                .add_schema(c_schema,
                    shape="circle",
                    center=["50%", "50%"],
                    radius="80%",
                    angleaxis_opts=opts.AngleAxisOpts(
                        min_=0,
                        max_=360,
                        is_clockwise=False,
                        interval=5,
                        axistick_opts=opts.AxisTickOpts(is_show=False),
                        axislabel_opts=opts.LabelOpts(is_show=False),
                        axisline_opts=opts.AxisLineOpts(is_show=False),
                        splitline_opts=opts.SplitLineOpts(is_show=False),
                    ),
                    radiusaxis_opts=opts.RadiusAxisOpts(
                        min_=0,
                        max_= round(max_v_set, 3),
                        interval= round(max_v_set/5,3),
                        splitarea_opts=opts.SplitAreaOpts(
                            is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                        ),
                    ),
                    polar_opts=opts.PolarOpts(),
                    splitarea_opt=opts.SplitAreaOpts(is_show=False),
                    splitline_opt=opts.SplitLineOpts(is_show=False),
                )
                .add(add_name1, add_value1, color="#f9713c")
                .add(add_name2, add_value2,  color="#b3e4a1", areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            )

            return c













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
    快速浏览输出到excel
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
        """
        搜索极端值并进行简单报告  
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


