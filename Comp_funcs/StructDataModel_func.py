#coding=utf-8
# python3.6
# Author: Scc_hy
# Create_Date: 2020-03-27
# function：特征筛选 和 模型训练
import numpy as np
import pandas as pd 
from datetime import datetime
from tqdm import tqdm
import sys, os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import copy
from EDA_func.Explore_func import tr_te_cols_distribute

def get_now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')



class Feats_filter():
    """
    特征筛选  
    1- 依据相关性剔除其中一个   corr_model_filter  
    2- 依据偏度以及峰度剔除   skew_kurt_filter
    3- corr_nunique特征筛选要增加 待增加  
       例：
         test_flg = df_all.label.isna()
         cols_lst = model_input_col
         df_skew = tr_te_cols_distribute(df_all, cols_lst, test_flg, func = 'skew')
         drop_skew_col = df_skew[(df_skew.tr_skew - df_skew.te_skew).map(abs) /  df_skew.te_skew.map(lambda x: x if x != 0 else 1) > 0.55].index.tolist()
         print(f'May drop {len(drop_skew_col)} cols')
         
         df_kurt = df_nunique = tr_te_cols_distribute(df_all, cols_lst, test_flg, func = 'kurt')
         drop_kurt_col = df_kurt[(df_kurt.tr_kurt - df_kurt.te_kurt).map(abs) /  df_kurt.te_kurt.map(lambda x: x if x != 0 else 1) > 0.7].index.tolist()
         print(f'May drop {len(drop_kurt_col)} cols')
         drop_kurt_skew = [i for i in  drop_kurt_col if i in drop_skew_col]
         print(f'May drop {len(drop_kurt_skew)} cols') 
    """
    def pd_get_col_corr(self, df: pd.DataFrame, corr_thoe: float) -> list:
        """
        param df: pd.DataFrame
        param corr_thoe: sparkdf.count()
        """
        corr_result = []
        corr = df.corr()
        col_name = corr.columns
        loop_n = corr.shape[0]
        for i in range(loop_n): 
            value_col = corr.loc[col_name[i]].values
            for j in range(i, loop_n): # 只对一半的进行遍历
                if j == i:
                    continue
                if abs(float(value_col[j])) > abs(float(corr_thoe)):
                    msg = "{},{},{}".format(col_name[i], col_name[j], value_col[j])
                    corr_result.append(msg)
        return corr_result

    def corr_model_filter(self, lgb_param, tr_dt, train_col, target ,col_corr):
        """
        param lgb_param: lgb参数 
        param tr_dt: 数据集  
        param train_col: 进入训练的特征  list 
        param target: 目标特征   str
        param col_corr: 根据get_col_corr 获取的高相关性列表   

        - 剔除相关性较高的其中一个特征
        例如：
          slt_param = {
            'bagging_freq': 5, 
            'boost_from_average':'false',
            'boost': 'gbdt',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_data_in_leaf': 50,
            'min_sum_hessian_in_leaf': 10.0,
            'tree_learner': 'serial',
            'n_estimators': 800,
            'early_stopping_rounds' : 500,
            'lambda_l2': 1,
            'is_unbalance': True,
            'objective': 'multiclass', 
            'n_jobs':10,
            'num_class': 5,
            'verbosity': 1}

            col_corr =  pd_get_col_corr(df_all, 0.965)
            model_input_col, impt_dct = get_model_feature(slt_param, df_all.loc[ ~test_flg, :], train_col, 'label' ,col_corr)
        """
        k_folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=2020)
        x_train=tr_dt[train_col].values
        y_train=tr_dt[target].values
        importances_ = np.array([0]*len(train_col), dtype= float)
        # 拟合
        for fold_i, (tr_index, te_index) in enumerate(k_folds.split(x_train, y_train)):
            print(f'Fold {fold_i + 1},start at: {get_now()}')
            x_tr, x_te = x_train[tr_index],x_train[te_index]
            y_tr, y_te = y_train[tr_index],y_train[te_index]
            train_data = lgb.Dataset(x_tr, label = y_tr)
            test_data = lgb.Dataset(x_te, label = y_te)
            
            clf = lgb.train(lgb_param, train_data
                            , valid_sets=[train_data, test_data]
                            , verbose_eval = 200)
            importances_ += clf.feature_importance(importance_type='gain')
            print(f'Get {fold_i + 1} times importances')                      
        
        importances_ /= (fold_i + 1)
        # 获得特征与重要性
        impt_dct = dict(zip(train_col, list(importances_)))
        model_input_col = copy.deepcopy(train_col)
        print('开始排除特征')
        for item in col_corr:
            temp = item.split(',')
            if temp[0] in  impt_dct.keys() and temp[1] in impt_dct.keys():
            if float(impt_dct[temp[0]]) > float(impt_dct[temp[1]]):
                del_col = temp[1] 
            else:
                del_col = temp[0]
            try :
                model_input_col.remove(del_col)
                print(f'remove {del_col}')
            except:
                pass
        return model_input_col, impt_dct
   
   def  skew_kurt_filter(df_all:pd.DataFrame , cols_lst:list, test_flg: np.array
                         , skew_filter = 0.55 , kurt_filter = 0.7 ) -> list:
        """
        依据偏度和峰值筛选特征
        : param df_all: pd.DataFrame   
        : param cols_lst: list
        : param test_flg: df_all.target.isna() 
        : param skew_filter: 测试集训练集的skew差 / 测试集skew   的比例阈值  
        : param kurt_filter: 测试集训练集的kurt差 / 测试集kurt   的比例阈值  
        """
        test_flg = df_all.label.isna()
        df_skew = tr_te_cols_distribute(df_all, cols_lst, test_flg, func = 'skew')
        drop_skew_col = df_skew[(df_skew.tr_skew - df_skew.te_skew).map(abs) /  df_skew.te_skew.map(lambda x: x if x != 0 else 1) > skew_filter].index.tolist()
        
        df_kurt = df_nunique = tr_te_cols_distribute(df_all, cols_lst, test_flg, func = 'kurt')
        drop_kurt_col = df_kurt[(df_kurt.tr_kurt - df_kurt.te_kurt).map(abs) /  df_kurt.te_kurt.map(lambda x: x if x != 0 else 1) > kurt_filter].index.tolist()
        drop_kurt_skew = [i for i in  drop_kurt_col if i in drop_skew_col]
        print(f'May drop {len(drop_kurt_skew)} cols') 
        return drop_kurt_skew
    
    
  

class models_func():
    """
    :classlgb_tr_model: 分类lgb模型
    """
    def classlgb_tr_model(x_train, y_train,  kf, lgb_param, feature_names, X_test = None
                , pred_flg = False, feval=None):
        """
        : param x_train: np.array  
        : param y_train: np.array  
        : param kf: StratifiedKFold(n_splits=3, shuffle=True, random_state=2020)    
        : param lgb_param: dict lgb的参数  
        : param feature_names: list 进入模型的特征名称 用于记录重要性  
        : param X_test: 预测集  
        : param pred_flg: bool  
        : param feval: lgb的自定义评估函数  
        例：
            model_input_col_final =  list(set(model_input_col) - set(drop_kurt_skew))
            folds_tr = StratifiedKFold(n_splits=3, shuffle=True, random_state=2020)
            tr_params = {
                'bagging_freq': 5, 
                'boost_from_average':'false',
                'boost': 'gbdt',
                'learning_rate': 0.05, #0.01,
                'max_depth': 12,#5,
                'min_data_in_leaf': 50,
                'min_sum_hessian_in_leaf': 10.0,
                'tree_learner': 'serial',
                'n_estimators': 2600,
                'early_stopping_rounds' : 500,
                'lambda_l2': 1,
                'objective': 'multiclass', 
                'n_jobs':10,
                'num_class': 5,
                'verbosity': 1}

            models, preds_test, importances, test_loss = tr_model(
                    x_train = tr_dt[model_input_col_final].values
                    ,y_train = tr_dt['label'].values
                    ,kf = folds_tr
                    ,lgb_param = tr_params
                    ,feature_names = model_input_col_final
                ,X_test = prd_dt[model_input_col_final].values
                ,pred_flg = True
                ,feval = lgb_f1_comp)

        """
        models,train_loss,test_loss = [], [], []
        preds_test = np.zeros((len(X_test), 5), dtype=np.float)
        importances = pd.DataFrame()
        # 拟合
        for fold_i, (tr_index, te_index) in enumerate(kf.split(x_train, y_train)):
            print(f'Fold {fold_i + 1},start at: {get_now()}')
            x_tr, x_te = x_train[tr_index],x_train[te_index]
            y_tr, y_te = y_train[tr_index],y_train[te_index]
            train_data = lgb.Dataset(x_tr, label = y_tr)
            test_data = lgb.Dataset(x_te, label = y_te)
            
            
            if  feval == None:
                clf = lgb.train(lgb_param, train_data
                                , valid_sets=[train_data, test_data]
                                , verbose_eval = 200)
            else:
                clf = lgb.train(lgb_param, train_data
                                , valid_sets=[train_data, test_data]
                                , verbose_eval=200, feval=feval)
            models.append(clf)

            # 模型的特征的重要性
            imp_df = pd.DataFrame()
            imp_df['feature'] = feature_names
            imp_df['split'] = clf.feature_importance()
            imp_df['gain'] = clf.feature_importance(importance_type='gain')
            imp_df['fold'] = fold_i + 1

            importances = pd.concat([importances, imp_df], axis=0)
            val_pred = np.argmax(clf.predict(x_te), axis=1)
            tr_pred = np.argmax(clf.predict(x_tr), axis=1)
            f1_tr = f1_score(y_te, val_pred, average='macro')
            f1_te = f1_score(y_tr, tr_pred, average='macro')
            train_loss.append(f1_tr)
            test_loss.append(f1_te)
            print('{} val f1 train/test: {:0.7f} / {:0.7f} mean_f1: {:0.7f}'\
                .format(fold_i + 1, train_loss[-1], test_loss[-1], np.mean(test_loss)))

            if pred_flg:
                preds_test += clf.predict(X_test)[:]
            print('-' * 80)

        if pred_flg:
            print(f'Train: {train_loss}\nVal: {test_loss}')
            print('-' * 80)
            print('Train{0:0.5f}_Test{1:0.5f}\n\n'.format(
                np.mean(train_loss), np.mean(test_loss)))
            preds_test /= (fold_i + 1)
            return models, preds_test, importances, np.mean(test_loss)
        else:
            return models, importances, np.mean(test_loss)
   
