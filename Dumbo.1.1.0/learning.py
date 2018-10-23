# -*- coding: utf-8 -*-
"""
changelog:

1.2.0  20181021
    ### 增加/自定义训练样本集
1.1.0  20181018  
    ### 扩展到下买单
    ### 行情数据按照RB日盘分3个交易时段，每个交易时段分开训练
    ### 最近的一段时间不参与训练，避免未来函数（标签）
1.0.0  20181009  first version
"""

from feature_extractor import FeatureExtractor
import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

import traceback


class ModelSelection(FeatureExtractor):
    """docstring for ModelSelection"""
    def __init__(self, *args):
        super(ModelSelection, self).__init__(*args)
        self.models = {
                        'RandomForestClassifier': RandomForestClassifier(random_state=0),
                         'ExtraTreesClassifier': ExtraTreesClassifier(random_state=0), 
                         'AdaBoostClassifier': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0), 
                         'GradientBoostingClassifier': GradientBoostingClassifier(random_state=0), 
                         # 'SVC': SVC(probability=True, random_state=0)
                         }
        self.model_grid_params = {
                                    'RandomForestClassifier': 
                                    {'max_features': [None], 'n_estimators': [10], 'max_depth': [10], 'min_samples_split': [2], 'criterion': ['entropy'], 'min_samples_leaf': [3]}, 
                                    'ExtraTreesClassifier': 
                                    {'max_features': [None], 'n_estimators': [10], 'max_depth': [10], 'min_samples_split': [2], 'criterion': ['entropy'], 'min_samples_leaf': [3]}, 
                                    'AdaBoostClassifier': 
                                    {'base_estimator__criterion': ['entropy'], 'base_estimator__max_depth': [None], 'base_estimator__min_samples_leaf': [3], 'base_estimator__min_samples_split': [2], 'base_estimator__max_features': [None]}, 
                                    'GradientBoostingClassifier': 
                                    {'max_features': [None], 'n_estimators': [10], 'max_depth': [10], 'min_samples_split': [2], 'min_samples_leaf': [3], 'learning_rate': [0.1], 'subsample': [1.0]},
                                    # 'SVC': [{'kernel': ['rbf'], 'gamma': [0.1], 'C': [1]}, {'kernel': ['linear'], 'C': [1, 10]}]
                                 }

        self.keys = self.models.keys()
        self.grid = {}

        self.predict_values_perday = {'ask': {0: {}, 1: {}, 2:{}}, 'bid': {0: {}, 1: {}, 2:{}}}
        self.cv_acc_perday = {'ask': {0: {}, 1: {}, 2:{}}, 'bid': {0: {}, 1: {}, 2:{}}}
        self.acc_perday = {'ask': {0: {}, 1: {}, 2:{}}, 'bid': {0: {}, 1: {}, 2:{}}}
        self.fscore_perday = {'ask': {0: {}, 1: {}, 2:{}}, 'bid': {0: {}, 1: {}, 2:{}}}
        self.true_values_perday = {'ask': {}, 'bid': {}}
        self.prediction_perday = {'ask': {}, 'bid': {}}
        self.timestamp = {'ask': {}, 'bid': {}}

        self.report = {'ask': {}, 'bid': {}}
        self.latest_sec = 0
        self.pred_sec = 0

    def clear_data_perday(self):

        for dr in ['ask', 'bid']:
            for ii in range(0, 3):
                for k in self.keys:
                    self.predict_values_perday[dr][ii][k] = []
                    self.cv_acc_perday[dr][ii][k] = []
                    self.acc_perday[dr][ii][k] = []
                    self.fscore_perday[dr][ii][k] = []
                self.true_values_perday[dr][ii] = []
                self.timestamp[dr][ii] = []

    def set_params(self, latest_sec, pred_sec):
        self.latest_sec = latest_sec
        self.pred_sec = pred_sec

    def get_sim_period(self, sd, ed):
        dd_list = []
        for root, sub_dirs, files in os.walk(os.path.join(self.proj_dir, "Features")):
            for s_file in files: 
                dd = int(s_file.split('.')[0][-8:])
                if (dd>=sd) and (dd<=ed):
                    dd_list.append(dd)  
        return np.unique(np.array(dd_list))

    def get_feature_data(self, fi, **kwargs):
        fn = "Features\\Features_" + fi + ".csv"
        f = os.path.join(self.proj_dir, fn)
        data = pd.read_csv(f, sep=',', header=None, index_col=False, **kwargs)

        return data.values

    def grid_fit(self, x_train, y_train, **kwargs):

        cv_acc = {}

        for key in self.keys:
            print("Running GridSearchCV for %s." % key)

            model = self.models[key]
            model_grid = self.model_grid_params[key]
            grid = GridSearchCV(model, model_grid, **kwargs)
            grid.fit(x_train, y_train)
            self.grid[key] = grid
            # self.cv_acc_perday[key][ss].extend([grid.best_score_] * length)
            cv_acc[key] = grid.best_score_

        return cv_acc

    def model_fit(self, x_train, y_train, x_test, y_test):

        acc = {}
        fscore = {}
        pred = {}

        for key in self.keys:
            print("Running training and testing for %s." % key)

            model = self.models[key]
            model.set_params(**self.grid[key].best_params_)
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)

            acc[key] = metrics.accuracy_score(y_test, prediction)
            fscore[key] = metrics.f1_score(y_test, prediction)
            pred[key] = prediction

        return acc, fscore, pred

    def pipline(self, day):

        print(day)
        self.clear_data_perday()

        for dr in ['ask', 'bid']:
            for it in range(0, 3):

                data = self.get_feature_data("%s_S%d_RB_%d" % (dr, it, day))

                # for ii in range(2400, self.latest_sec+self.traded_time+2*self.pred_sec, self.pred_sec):
                for ii in range(self.latest_sec+self.traded_time, data.shape[0]-self.pred_sec, self.pred_sec):
                    print('--------------------Rolling Window at Session %d, Prediction direction %s: Time = %s--------------------' % (it, dr, ii))
                    # 训练集，最近的traded_time不做训练集，否则用到未来标签
                    data_train = data[ii-self.latest_sec-self.traded_time: ii-self.traded_time, :]
                    data_train = data_train[data_train[:, 0]!=99, :]
                    x_train = data_train[:, 4:]
                    y_train = data_train[:, 0]

                    # 测试集
                    data_test = data[ii: ii+self.pred_sec, :]
                    data_test = data_test[data_test[:, 0]!=99, :]
                    x_test = data_test[:, 4:]
                    y_test = data_test[:, 0]

                    cv_acc = self.grid_fit(x_train, y_train, cv=5, scoring='accuracy')
                    acc, f_score, pred = self.model_fit(x_train, y_train, x_test, y_test)

                    # 记录时间戳（秒)，只推送了有效的预测集的时间戳，既可能间断
                    self.timestamp[dr][it].extend(list(data_test[:, 1]))
                    self.true_values_perday[dr][it].extend(list(data_test[:, 0]))

                    for k in self.keys:
                        self.cv_acc_perday[dr][it][k].extend([cv_acc[k]] * data_test.shape[0])
                        self.acc_perday[dr][it][k].extend([acc[k]] * data_test.shape[0])
                        self.fscore_perday[dr][it][k].extend([f_score[k]] * data_test.shape[0])
                        self.predict_values_perday[dr][it][k].extend(list(pred[k]))
        
                self.report[dr][it] = self.score_report(self.acc_perday[dr][it], self.fscore_perday[dr][it], sort_by='Accuracy_mean')
        
        return self.output_report(day)

    def score_report(self, acc, fscore, sort_by):

        report = pd.concat([
            pd.DataFrame([self.keys]), 
            pd.DataFrame([map(lambda x: np.mean(acc[x]), acc)]),
            pd.DataFrame([map(lambda x: np.std(acc[x]), acc)]),
            pd.DataFrame([map(lambda x: max(acc[x]), acc)]),
            pd.DataFrame([map(lambda x: min(acc[x]), acc)]),
            pd.DataFrame([map(lambda x: np.mean(fscore[x]), fscore)])
            ], axis=0).T

        report.columns = ['Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score']
        report.index.rename('Ranking', inplace=True)          
        
        return report.sort_values(by = [sort_by], ascending=False)

    def output_report(self, date):
        po = os.path.join(self.proj_dir, "Prediction")
        if not os.path.exists(po):
            os.mkdir(po)
        po = os.path.join(self.proj_dir, "Prediction", str(date))
        if not os.path.exists(po):
            os.mkdir(po)

        report_summary = []
        rank_index = []
        for dr in ['ask', 'bid']:
            report_summary_ = []
            acc_perday_ = {k: [] for k in self.keys}
            fscore_perday_ = {k: [] for k in self.keys}
            for ii in range(0, 3):
                # self.report[dr][ii].to_csv(fn+"%s_S%d_%d.csv" % (dr, ii, date), index=True, header=True)
                rank_index.extend(self.report[dr][ii].index.tolist())

                # report_summary_.extend(self.report[dr][ii].index.tolist())
                report_summary_.extend(self.report[dr][ii].values.tolist())
            
                for k in self.keys:
                    acc_perday_[k].extend(self.acc_perday[dr][ii][k])
                    fscore_perday_[k].extend(self.fscore_perday[dr][ii][k])

            s_combine = pd.concat([
                pd.DataFrame([self.keys]),
                pd.DataFrame([map(lambda x: np.mean(acc_perday_[x]), acc_perday_)]),
                pd.DataFrame([map(lambda x: np.std(acc_perday_[x]), acc_perday_)]),
                pd.DataFrame([map(lambda x: max(acc_perday_[x]), acc_perday_)]),
                pd.DataFrame([map(lambda x: min(acc_perday_[x]), acc_perday_)]),
                pd.DataFrame([map(lambda x: np.mean(fscore_perday_[x]), fscore_perday_)]),
                ], axis=0).T

            s_combine.columns = ['Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score']
            s_combine.index.rename('Ranking', inplace=True)       
            s_combine = s_combine.sort_values(by = ['Accuracy_mean'], ascending=False)

            rank_index.extend(s_combine.index.tolist())
            # report_summary_.extend(s_combine.index.tolist())
            report_summary_.extend(s_combine.values.tolist())
            report_summary.extend(report_summary_)

        report_summary_df = pd.DataFrame(report_summary, 
                                        index=[
                                        ['ask']*16+['bid']*16, 
                                        ['s1']*4+['s2']*4+['s3']*4+['s1+s2+s3']*4 + ['s1']*4+['s2']*4+['s3']*4+['s1+s2+s3']*4,
                                        rank_index,
                                        ], 
                                        columns=['Estimator','Accuracy_mean','Accuracy_std','Accuracy_max','Accuracy_min','F_score'])
        report_summary_df.to_excel(pd.ExcelWriter(os.path.join(po, "report_%d.xlsx" % date)), merge_cells=True)

        fn = os.path.join(po, "prediction_")
        for dr in ['ask', 'bid']:
            self.prediction_perday[dr] = pd.concat([
                pd.DataFrame([0]*len(self.timestamp[dr][0])+[1]*len(self.timestamp[dr][1])+[2]*len(self.timestamp[dr][2]), columns=['SS']),
                pd.DataFrame(self.timestamp[dr][0]+self.timestamp[dr][1]+self.timestamp[dr][2], columns=['timestamp_s']),
                pd.DataFrame(self.true_values_perday[dr][0]+self.true_values_perday[dr][1]+self.true_values_perday[dr][2], columns=['true_values']),
                pd.DataFrame({k+"_pv": self.predict_values_perday[dr][0][k]+\
                                       self.predict_values_perday[dr][1][k]+\
                                       self.predict_values_perday[dr][2][k] for k in self.keys}),
                pd.DataFrame({k+"_cv": self.cv_acc_perday[dr][0][k]+\
                                       self.cv_acc_perday[dr][1][k]+\
                                       self.cv_acc_perday[dr][2][k] for k in self.keys}),
                pd.DataFrame({k+"_acc": self.acc_perday[dr][0][k]+\
                                       self.acc_perday[dr][1][k]+\
                                       self.acc_perday[dr][2][k] for k in self.keys}),
                pd.DataFrame({k+"_F": self.fscore_perday[dr][0][k]+\
                                       self.fscore_perday[dr][1][k]+\
                                       self.fscore_perday[dr][2][k] for k in self.keys}),
                ], axis=1)

            self.prediction_perday[dr].to_csv(fn+"%s_%d.csv" % (dr, date), index=False, header=True)
  
        color_ = ['r','orange','y','g','b']
        for dr in ['ask', 'bid']:
            sns.set_style("whitegrid")
            plt.figure(figsize=(18,6))
            
            for key in self.keys:
                plt.plot(self.prediction_perday[dr].loc[:, key+"_acc"], '-o', label=key, lw=1, markersize=3)
                plt.legend(loc=0)

            plt.ylim(-0.5, 1.5)
            plt.legend(loc=0)
            plt.xlabel('Rolling Window Numbers', size=15)
            plt.ylabel('Accuracy', size=15)
            plt.savefig(po+"\\"+"%s accuracy time series.png" % dr)

            sns.set_style("whitegrid")
            plt.figure(figsize=(18,6))

            for index, key in enumerate(self.keys):
                plt.plot(self.prediction_perday[dr].loc[:, key+"_cv"], '-o', label=key, lw=1, color=color_[index], markersize=3)
                plt.legend(loc=3)

            plt.xlabel('Rolling Window Numbers', size=15)
            plt.ylabel('CV Mean Accuracy', size=15)
            plt.savefig(po+"\\"+"%s cross validation.png" % dr)

        return self.prediction_perday

if __name__ == "__main__":

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.split(proj_dir)[0], "data\\stock index future")
    start_date = 20170103
    end_date = 20170307

    t1 = time.clock()

    modelselect = ModelSelection(proj_dir, data_dir, 600)
    modelselect.set_params(30*60, 10)

    period = modelselect.get_sim_period(start_date, end_date)
    # period = np.array([20170103, 20170104, 20170214])

    for day in period:
        try:
            modelselect.pipline(day)
        except Exception as e:
            traceback.print_exc()

    print(u"总耗时 %.2f sec" % (time.clock()-t1))
