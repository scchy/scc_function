# python 3.6
# author: Scc_hy 
# create date: 2020-03-03
# Function： 挖掘测试

from datetime import datetime
import time
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import missingno as msn
import matplotlib.pyplot as plt
from scc_function.progressing import My_Progress


def get_col_uniq(tr_dt):
    col_uniq= {}
    for col in tr_dt.columns:
        col_uniq[col] = tr_dt.loc[:,col].nunique()
    col_uniq_sorted = sorted(col_uniq.items(), key= lambda c:c[1])
    return col_uniq_sorted



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


def num_cat_scan(df, num_col_in = None, scanmethod='5sd'):
    """搜索极端值并进行简单报告  
    scanmethod: 极端值确认标准， 5sd 超过均数+-5被标准差  
    返回: 包括全部数值变量搜索结果的数据框  
    """
    # 数值型遍历
    dfres = pd.DataFrame(
        columns=['记录数', '均值', '标准差', '最小值', '最大值', '偏态','小值超界', '大值超界'])
    num_col = list(df.columns) if num_col_in == None else num_col_in
    p = My_Progress(num_col, width=45)
    for col in num_col:
        msg = p.progress()
        print('目前处理(数值特征): {0:{2}<10}, 进度: {1}'.format(
            col, msg, chr(12288)))
        try:
            std_m = int(scanmethod[:-2])
            cnt = len(df[col])
            v_std = df[col].std()
            v_mean = df[col].mean()
            v_skew = df[col].skew()
            dfres.loc[col, :] = [cnt, v_mean, v_std,
                                 df[col].min(), df[col].max(), v_skew,
                                sum(df[col] < v_mean - std_m * v_std),
                                sum(df[col] > v_mean + std_m * v_std)
                                ]
        except:  # 非数值变量
            print('{}无法分析'.format(col))
    dfres.reset_index(inplace=True)
    return dfres



def run_oof(lgb_param, X_train, y_train,  
            kf, eval_fun, n_fold, feature_names
            , X_test=None, istest=False, ratio=1, feval=None):
    models = []
    preds_train = np.zeros((len(X_train), 3), dtype=np.float)
    train_loss = []
    importances = pd.DataFrame()

    if istest:
        preds_test = np.zeros((len(X_test), 3), dtype=np.float)
        test_loss = []

    for fold_n, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        print('Fold', fold_n + 1, 'started at：',
              datetime.now().strftime('%Y-%m-%d_%H:%M'))

        x_tr, x_te = X_train[train_index], X_train[test_index]
        y_tr, y_te = y_train[train_index], y_train[test_index]

        weights = [ratio if val == 1 else 1 for val in y_tr]

        train_data = lgb.Dataset(x_tr, label=y_tr,  weight=weights)
        valid_data = lgb.Dataset(x_te, label=y_te)
        if feval == None:
            clf = lgb.train(lgb_param, train_data, valid_sets=[
                            train_data, valid_data], verbose_eval=400)
        else:
            clf = lgb.train(lgb_param, train_data, valid_sets=[
                            train_data, valid_data], verbose_eval=400, feval=feval)
        # eval_set = [(x_te, y_te)]
        # 记录每个模型
        models.append(clf)
        if istest:
            train_loss.append(eval_fun(y_tr, np.argmax(
                clf.predict(x_tr)[:], 1), average='macro'))

        # 模型的特征的重要性
        imp_df = pd.DataFrame()
        imp_df['feature'] = feature_names
        imp_df['split'] = clf.feature_importance()
        imp_df['gain'] = clf.feature_importance(importance_type='gain')
        imp_df['fold'] = fold_n + 1

        importances = pd.concat([importances, imp_df], axis=0)

        if istest:
            test_loss.append(eval_fun(y_te, np.argmax(
                clf.predict(x_te)[:], 1), average='macro'))
            preds_train[test_index] = clf.predict(x_te)[:]
            preds_test += clf.predict(X_test)[:]
            print('{0}: Train {1:0.7f} Val {2:0.7f}/{3:0.7f}'.format(fold_n + 1,
                                                                     train_loss[-1], test_loss[-1], np.mean(test_loss)))
        print('-' * 80)

    if istest:
        print('Train: ', train_loss, '\nVal: ', test_loss)
        print('-' * 80)
        print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(
            np.mean(train_loss), np.mean(test_loss)))
        preds_test /= n_fold
        return models, preds_train, preds_test, importances
    else:
        return models, importances

def check_tr_te_distr(train, test):
    """
    检查 测试集和训练集的分布   
    如果超过 平均0.55则分布有偏
    """
    train['label'] = 0
    test['label'] = 1
    test.columns = list(train.columns)
    trte = pd.concat([train, test], axis=0, ignore_index=True)
    trte.reset_index(drop = True, inplace = True)
    print(trte.index)
    tr_te_fold = 5
    folds = StratifiedKFold(n_splits=tr_te_fold, shuffle=True, random_state=42)
    tr_te_params = {
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
        'n_estimators': 1000,
        'early_stopping_rounds' : 500,
        'verbosity': 1}

    run_oof(lgb_param = tr_te_params,
            X_train=trte.iloc[:, :-1].values,
            y_train=trte['label'].values,
            kf=folds,
            eval_fun=f1_score,
            n_fold=tr_te_fold,
            feature_names=list(trte.iloc[:,:-1].columns),
            istest=False)

