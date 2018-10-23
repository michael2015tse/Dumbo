# -*- coding: utf8 -*-
"""
changelog:

1.1.0  20181018  
    ### 扩展到下买单
    ### 行情数据按照RB日盘分3个交易时段，每个交易时段分开训练
    ### 最近的一段时间不参与训练，避免未来函数（标签）
1.0.0  20181009  first version
"""


from learning import ModelSelection
import os, time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import traceback


class BackTesting(ModelSelection):
    """docstring for BackTesting"""
    def __init__(self, start_time, end_date, proj_dir, data_dir, traded_time):
        super(BackTesting, self).__init__(proj_dir, data_dir, traded_time)
        self.start_time = start_time
        self.end_date = end_date
        self.result = {
                        'ask': {'time': [], 'profit_persec': [], 'is_win_persec': [], 'cumprofit': [], 'profit_perbet': []},
                       'bid': {'time': [], 'profit_persec': [], 'is_win_persec': [], 'cumprofit': [], 'profit_perbet': []}
                        }
 
        self.prediction = {0: [], 1: [], 2: []}

        self.ask_1 = []
        self.bid_1 = []
        self.second_basic = []

    def clear_data_perday(self):

        super().clear_data_perday()  # 调用父类的方法
        self.result = {
                'ask': {'time': [], 'profit_persec': [], 'is_win_persec': [], 'cumprofit': [], 'profit_perbet': []},
               'bid': {'time': [], 'profit_persec': [], 'is_win_persec': [], 'cumprofit': [], 'profit_perbet': []}
                }

    def get_sim_period(self, sd, ed):
        f_dd_list = []
        for root, sub_dirs, files in os.walk(os.path.join(self.proj_dir, "Features")):
            for s_file in files: 
                dd = int(s_file.split('.')[0][-8:])
                if (dd>=sd) and (dd<=ed):
                    f_dd_list.append(dd)  

        p_dd_list = []
        for root, sub_dirs, files in os.walk(os.path.join(self.proj_dir, "Prediction")):
            for s_sub in sub_dirs: 
                dd = int(s_sub)
                if (dd>=sd) and (dd<=ed):
                    p_dd_list.append(dd)  

        return np.array([day for day in p_dd_list if f_dd_list])

    def trainning_and_test(self, day, dr):
        # 训练-预测
        f = os.path.join(self.proj_dir, "Prediction\\"+str(day)+"\\"+"prediction_%s_%d.csv" % (dr, day))
        if os.path.exists(f):
            prediction = pd.read_csv(f, sep=',', header=0)
        else:
            prediction = self.pipline(day)

        for ii in range(0, 3):
            self.prediction[ii] = prediction.loc[prediction.loc[:, 'SS']==ii, :].reset_index(drop=True)

    def get_snapshot_data(self, fn, **kwargs):
        f = os.path.join(self.data_dir, fn + ".csv")
        table = pd.read_csv(f, sep=',', header=0, index_col=False, **kwargs)

        updatetime = table.iloc[:, 2].values.astype(str)
        self.ask_1 = table.iloc[:, 7].values.astype(np.float32)
        self.bid_1 = table.iloc[:, 8].values.astype(np.float32)

        self.ask_1 = np.where(np.isnan(self.ask_1), 0, self.ask_1)
        self.bid_1 = np.where(np.isnan(self.bid_1), 0, self.bid_1)

        self.second_basic = \
            np.array([float(updatetime[ii][0:2]) * 3600 + float(updatetime[ii][3:5]) * 60
                      + float(updatetime[ii][6:8]) - 32400.0 for ii in range(0, updatetime.shape[0])])

    def trade_action(self, bt, et, prediction):
        """
        :param bt: 开盘
        :param et: 收盘
        :param prediction: 预测结果df
        :return: 交易时段长度的动作列表
        """

        sub, tr_value = [], []
        keys_list = list(self.keys)
        index = 0
        for t in range(bt, et):

            if index < prediction.shape[0] and t == prediction.loc[index, 'timestamp_s']:
                if t <= et - self.traded_time:
                    cv_acc = prediction.loc[index, [k + "_cv" for k in keys_list]].values
                    best_model = keys_list[np.argmax(cv_acc)]
                    sub.append(prediction.loc[index, best_model+"_pv"])
                else:
                    # 收盘前self.traded_time不开仓，给足时间让之前的交易在最后的self.traded_time内解决
                    sub.append(0)
                tr_value.append(prediction.loc[index, "true_values"])
                index += 1
            else:
                sub.append(99)
                tr_value.append(99)

        return sub, tr_value

    def open_price(self, bid, ask, dr):
        return bid if dr == 'ask' else ask

    def close_price(self, bid, ask, dr):
        return ask if dr == 'ask' else bid

    def calc_profit(self, st, et, sub, tr_v, dr, how='c'):

        if how == 'c':
            return self.calc_profit_c(st, et, sub, tr_v, dr)
        else:
            return self.calc_profit_o(st, et, sub, tr_v, dr)

    def calc_profit_o(self, st, et, sub, tr_v, dr):
        
        index = -1
        profit_persec = [0] * len(sub)
        is_win_persec = [0] * len(sub)

        cond = (self.second_basic<=et) & (self.second_basic>=st)
        second_basic = self.second_basic[cond]
        ask = self.ask_1[cond]
        bid = self.bid_1[cond]

        for ii in range(0, et-st):
            act = sub[ii]
            trv = tr_v[ii]

            if ii != 0:
                index_array = np.where((second_basic>=ii+st) & (second_basic<ii+st+1))[-1]
            else:
                index_array = np.where(second_basic<=ii+st)[-1]

            if len(index_array)>0:
                index = index_array[-1]

            if index >= 0:
                if act == trv == 1:
                    op = self.open_price(bid[index], ask[index], dr)
                    index_min = np.where(second_basic<=ii+st+self.traded_time)[0][-1]
                    for i_ in range(index, index_min):
                        cp = self.close_price(bid[i_], ask[i_], dr)
                        if dr == 'ask':
                            if cp < op:                           
                                profit_persec[ii] = op - cp
                                is_win_persec[ii] = 1
                                break
                        else:
                            if cp > op:
                                profit_persec[ii] = cp - cp
                                is_win_persec[ii] = 1
                                break
                elif act != trv:
                    if act == 1:
                        index_min = np.where(second_basic<=ii+st+self.traded_time)[0][-1] + 1
                        index_min = index_min if index_min < len(ask) else -1
                        op = self.open_price(bid[index], ask[index], dr)
                        cp = self.close_price(bid[index_min], ask[index_min], dr)
                        profit_persec[ii] = op - cp if dr == 'ask' else cp - op
                        is_win_persec[ii] = -1

        return profit_persec, is_win_persec, profit_persec

    def calc_profit_c(self, st, et, sub, tr_v, dr):

        index = -1
        profit_persec = [0] * len(sub)  # 每秒的收益，记录在平仓位置
        profit_perbet = [0] * len(sub)  # 每笔开仓的收益，记录在开仓位置
        is_win_persec = [0] * len(sub)

        cond = (self.second_basic <= et) & (self.second_basic >= st)
        second_basic = self.second_basic[cond]
        ask = self.ask_1[cond]
        bid = self.bid_1[cond]

        for ii in range(0, et-st):

            act = sub[ii]
            trv = tr_v[ii]

            if ii != 0:
                index_array = np.where((second_basic>=ii+st) & (second_basic<ii+st+1))[-1]
            else:
                index_array = np.where(second_basic<=ii+st)[-1]

            if len(index_array)>0:
                index = index_array[-1]

            if index >= 0:
                if act == trv == 1:
                    op = self.open_price(bid[index], ask[index], dr)
                    index_min = np.where(second_basic<=ii+st+self.traded_time)[0][-1]
                    for i_ in range(index, index_min):
                        cp = self.close_price(bid[i_], ask[i_], dr)
                        if dr == 'ask':
                            if cp < op:
                                i_c = int(second_basic[i_]-st-1)
                                i_c = i_c if i_c >= 0 else 0
                                profit_persec[i_c] += op - cp
                                profit_perbet[ii] = op - cp
                                is_win_persec[ii] = 1
                                break
                        else:
                            if cp > op:
                                i_c = int(second_basic[i_]-st-1)
                                i_c = i_c if i_c >= 0 else 0
                                profit_persec[i_c] += cp - op
                                profit_perbet[ii] = cp - op
                                is_win_persec[ii] = 1
                                break

                elif act != trv:
                    if act == 1:
                        # 若做错，在下一秒平仓或收盘价平仓
                        index_min = np.where(second_basic<=ii+st+self.traded_time)[0][-1] + 1
                        index_min = index_min if index_min < len(ask) else -1
                        i_c = int(second_basic[index_min]-st) - 1
                        i_c = i_c if i_c >= 0 else 0
                        op = self.open_price(bid[index], ask[index], dr)
                        cp = self.close_price(bid[index_min], ask[index_min], dr)
                        if dr == 'ask':
                            profit_persec[i_c] += op - cp
                            profit_perbet[ii] = op - cp
                        else:
                            profit_persec[i_c] += cp - op
                            profit_perbet[ii] = cp - op
                        is_win_persec[ii] = -1

        return profit_persec, is_win_persec, profit_perbet

    def running(self, tr_period, pr_period, day):

        print(day)
        self.clear_data_perday()
        self.set_params(tr_period, pr_period)

        for dr in ['ask', 'bid']:

            self.trainning_and_test(day, dr)
            self.get_snapshot_data(os.path.join("Data_by_Day", "RB_"+str(day)))

            sess = [[0, 4500], [5400, 9000], [16200, 21600]]

            for ii in range(0, 3):
                # 根据预测结果生成执行动作列表
                sub, trv = self.trade_action(sess[ii][0], sess[ii][1], self.prediction[ii])
                # 计算策略收益
                profit, is_win, profit_perbet = self.calc_profit(sess[ii][0], sess[ii][1], sub, trv, dr, 'c')

                self.result[dr]['time'].extend(list(range(sess[ii][0], sess[ii][1])))
                self.result[dr]['is_win_persec'].extend(is_win)
                self.result[dr]['profit_persec'].extend(profit)
                self.result[dr]['profit_perbet'].extend(profit_perbet)

            self.result[dr]['cumprofit'] = np.cumsum(self.result[dr]['profit_persec'])

        self.output_performance(day)

    def output_performance(self, date):
        po = os.path.join(self.proj_dir, "Backtest")
        if not os.path.exists(po):
            os.mkdir(po)
        po = os.path.join(self.proj_dir, "Backtest", str(date))
        if not os.path.exists(po):
            os.mkdir(po)

        writer = pd.ExcelWriter(os.path.join(po, "pnl_%d.xlsx" % date))
        for dr in ['ask', 'bid']:
            pd.DataFrame(self.result[dr]).to_excel(writer, sheet_name=dr)
        writer.save()

        sns.set_style("whitegrid")
        plt.figure(figsize=(18, 6))
        color_ = {'ask': '#ff7f0e', 'bid': '#aec7e8'}
        for dr in ['ask', 'bid']:        
            plt.plot(self.result[dr]['time'], self.result[dr]['cumprofit'], lw=3, color=color_[dr], label=dr)
        plt.legend(loc=0)
        plt.xlabel('Time (s)', size=15)
        plt.ylabel('Profit & Loss', size=15)
        plt.savefig(po + "\\" + "backtest_perf_%d.png" % date)

        for dr in ['ask', 'bid']:
            sns.set_style("whitegrid")
            fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))
            plt.plot(self.result[dr]['time'], self.result[dr]['cumprofit'], linewidth=3, color=color_[dr])
            ax1.set_ylabel('Profit & Loss', size=15)
            ax1.set_xlabel('Time (s)', size=15)

            ax2 = ax1.twinx()
            plt.bar(self.result[dr]['time'], self.result[dr]['profit_persec'], width=30, color=color_[dr])
            ax2.set_ylabel('Profit per Sec', size=15)
            ax2.set_xlabel('Time (s)', size=15)

            plt.savefig(po + "\\" + "backtest_perf_%s_%d.png" % (dr, date))

            sns.set_style("whitegrid")
            fig, ax1 = plt.subplots(1, 1, figsize=(18, 6))
            plt.plot(self.second_basic, self.ask_1, color='#ff7f0e', lw=1, label='ask')
            plt.plot(self.second_basic, self.bid_1, color='#aec7e8', lw=1, label='bid')
            plt.legend(loc=0)
            ax1.set_ylabel('Price', size=15)
            ax1.set_xlabel('Time (s)', size=15)

            ax2 = ax1.twinx()
            plt.bar(self.result[dr]['time'], self.result[dr]['profit_perbet'], width=30, color=color_[dr])
            ax2.set_ylabel('Profit per %s' % dr, size=15)
            ax2.set_xlabel('Time (s)', size=15)

            plt.savefig(po + "\\" + "Profit_per_%s_%d.png" % (dr, date))


if __name__ == "__main__":

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.split(proj_dir)[0], "data\\stock index future")
    start_date = 20170105
    end_date = 20170307

    t1 = time.clock()

    backtesting = BackTesting(start_date, end_date, proj_dir, data_dir, 600)

    period = backtesting.get_sim_period(start_date, end_date)

    for day in period:
        try:
            backtesting.running(1800, 10, day)
        except Exception as e:
            traceback.print_exc()

    print(u"总耗时 %.2f sec" % (time.clock()-t1))