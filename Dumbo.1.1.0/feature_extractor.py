# -*- coding: utf-8 -*-
"""
changelog:

1.1.0  20181018  
    ### 补充买单特征和标签，扩展到下买单
    ### 行情数据按照RB日盘分3个交易时段，每个交易时段分开标签
1.0.0  20181009  first version
"""


from train_test_builder import TrainTestBuilder
import os
import time
import numpy as np
import pandas as pd


class FeatureExtractor(TrainTestBuilder):
    """docstring for FeatureExtractor"""
    def __init__(self, pd, dd, traded_time):
        super(FeatureExtractor, self).__init__(dd)
        self.proj_dir = pd
        self.traded_time = traded_time

    def data_prepare(self, fn):
        # 提取基础数据
        self.data_builder(fn)
        # 计算开盘后的rise_ask序列，可以处理休市的时间段
        rise_ratio_ask = []
        for ii in range(0, 30):
            before_time = 60.0 * 6 + 30 * ii
            rise_ratio = self.rise_price(self.ask_1, self.second_basic, before_time)
            rise_ratio_ask.append(rise_ratio)

        rise_ratio_bid = []
        for ii in range(0, 30):
            before_time = 60.0 * 6 + 30 * ii
            rise_ratio = self.rise_price(self.bid_1, self.second_basic, before_time)
            rise_ratio_bid.append(rise_ratio)

        # 计算加权量
        # ask_vol_1 = self.ask_vol_1
        ask_vol_1 = self.ask_vol_1[self.second_basic>=0] # TSE：调整使与rise_ratio对齐，都从上午开盘开始截断
        ask_vol_2 = np.full(ask_vol_1.shape, 0, dtype=np.float32)
        ask_vol_3 = ask_vol_2.copy()
        # bid_vol_1 = self.bid_vol_1
        bid_vol_1 = self.bid_vol_1[self.second_basic>=0]
        bid_vol_2 = np.full(bid_vol_1.shape, 0, dtype=np.float32)
        bid_vol_3 = bid_vol_2.copy()
        
        w_ab_100, w_a_b_100 = \
        self.weight_percentage(wt=[1.0, 0.0, 0.0], av_1=ask_vol_1, bv_1=bid_vol_1, 
                                                   av_2=ask_vol_2, bv_2=bid_vol_2,
                                                   av_3=ask_vol_3, bv_3=bid_vol_3)
        w_list = [w_ab_100, w_a_b_100]
        return w_list, rise_ratio_ask, rise_ratio_bid

    def weight_percentage(self, wt, av_1, av_2, av_3, bv_1, bv_2, bv_3):
        w1, w2, w3 = wt
        w_ask = w1 * av_1 + w2 * av_2 + w3 * av_3
        w_bid = w1 * bv_1 + w2 * bv_2 + w3 * bv_3

        w_ab = np.where(w_bid==0, 0, w_ask / w_bid)
        w_a_b = np.where(w_ask+w_bid==0, 0, (w_ask - w_bid) / (w_ask + w_bid))

        return w_ab, w_a_b

    def rise_price(self, price, time_second, before_time):
        """
        返回序列，时间戳不大于before_time时，返回当前时间戳的涨跌幅；
        大于before_time后，返回窗口before_time的涨跌幅 
        """
        price = price[time_second>=0]
        time_second = time_second[time_second>=0]
        price = np.where(price==0, np.mean(price), price)
        # rise_ratio = []
        # index = np.where(time_second>=before_time)[0][0]
        # for ii in range(0, index):
        #     rise_ratio.append(price[ii]*100.0/price[0] - 100.0)

        # index_start = 0
        # for ii in range(index, len(price)):
        #     while time_second[index_start]<time_second[ii]-before_time:
        #         index_start+=1
        #     rise_ratio.append(price[ii]*100.0/price[index_start] - 100.0)

        index_start = 0
        rise_ratio = [0]
        for ii in range(1, len(price)):
            while time_second[index_start]<time_second[ii]-before_time:
                index_start+=1
            rise_ratio.append(price[ii]*100.0/price[index_start] - 100.0)

        return rise_ratio

    def is_down(self, p0, parray):
        # 未来一段时间的卖价低于当前买价，说明未来下跌
        return 1 if p0 > np.min(parray) else 0

    def is_up(self, p0, parray):
        # 未来一段时间的买价高于当前卖价，说明未来上涨
        return 1 if p0 < np.max(parray) else 0

    def label_generator(self, str_time, end_time, traded_time):
        
        is_down = []
        is_up = []

        cond = (self.second_basic <= end_time) & (self.second_basic >= str_time)
        second_basic = self.second_basic[cond]
        ask = self.ask_1[cond]
        bid = self.bid_1[cond]

        index = -1
        for ii in range(str_time, end_time):
            if ii == str_time: # 3个交易时段，开盘时间分别为9:00， 10:30， 13:30
                index_array = np.where(second_basic<=ii)[-1]  # np.where()的结果是一个长度为1的tuple
            else:
                index_array = np.where((second_basic>=ii) & (second_basic<ii+1))[-1]

            if len(index_array) > 0:
                index = index_array[-1]

            if index >= 0:
                if ii < second_basic[-1] - traded_time:
                    index_end = np.where(second_basic <= ii + traded_time)[0][-1]
                    if index_end > index:

                        is_down.append(self.is_down(bid[index], ask[index: index_end]))
                        is_up.append(self.is_up(ask[index], bid[index: index_end]))

                    else:
                        is_up.append(99)
                        is_down.append(99)

                else:
                    # 临近本段交易时段的收盘
                    is_down.append(self.is_down(bid[index], ask[index:]))
                    is_up.append(self.is_up(ask[index], bid[index:]))

            else:
                # 第一次进入循环就缺数据，此时index=-1
                is_up.append(99)
                is_down.append(99)

        return is_up, is_down

    def feature_extractor(self, str_time, end_time, w_list, rise_ratio_list):
        # 根据tick数据，特征按照秒频率给
        rise_ratio_second = [list() for i in range(len(rise_ratio_list))]
        w_divid = [list() for i in range(int(len(w_list)/2))]
        w_diff = [list() for i in range(int(len(w_list)/2))]
        ask_p = []
        bid_p = []
        ask_1 = self.ask_1[self.second_basic>=0]
        bid_1 = self.bid_1[self.second_basic>=0]

        index_one = np.where(self.second_basic<=0)[0][-1] # 上午开盘第0秒最后一个tick的index

        index = -1
        for ii in range(str_time, end_time):
            if ii == str_time:
                index_array = np.where(self.second_basic<=ii)[-1]
            else:
                index_array = np.where((self.second_basic>=ii) & (self.second_basic<ii+1))[-1]

            if len(index_array) > 0:
                index = index_array[-1] - index_one

            if index >= 0:
                for ix in range(0, len(rise_ratio_list)):
                    # 开盘后第index个tick的第ix种窗口涨跌幅，index是1s的最后一个tick，涨跌幅是用第一个tick计算的
                    rise_ratio_second[ix].append(rise_ratio_list[ix][index])
                for ix in range(0, len(w_divid)):
                    w_divid[ix].append(w_list[ix][index])
                for ix in range(0, len(w_diff)):
                    w_diff[ix].append(w_list[ix*2+1][index])
                ask_p.append(ask_1[index])
                bid_p.append(bid_1[index])

            else:
                for ix in range(0, len(rise_ratio_list)):
                    rise_ratio_second[ix].append(0.0)
                for ix in range(0, len(w_divid)):
                    w_divid[ix].append(0.0)
                for ix in range(0, len(w_diff)):
                    w_diff[ix].append(0.0)
                ask_p.append(0.0)
                bid_p.append(0.0)

        return rise_ratio_second, w_divid, w_diff, ask_p, bid_p

    def output_data_set(self, fn, pi, po):
        po = os.path.join(self.proj_dir, po)
        if not os.path.exists(po):
            os.mkdir(po)
        trade_session = [[0, 4500], [5400, 9000], [16200, 21600]]  # 9:00~10:15  10:30~11:30  13:30~15:00

        w_list, rise_ratio_ask, rise_ratio_bid = self.data_prepare(os.path.join(pi, fn))
        rise_ratio_ab = {'ask': rise_ratio_ask, 'bid': rise_ratio_bid}

        # 标签
        label = {'ask': [[], [], []], 'bid': [[], [], []]}  # 分3个交易时段
        for it in range(0, 3):
            label['bid'][it], label['ask'][it] = \
                self.label_generator(trade_session[it][0], trade_session[it][1], self.traded_time)

        # 特征
        rise_ratio_second = {'ask': [[], [], []], 'bid': [[], [], []]}
        w_divid = [[], [], []]
        w_diff = [[], [], []]
        ask = [[], [], []]
        bid = [[], [], []]

        for k in rise_ratio_second.keys():
            for t in range(0, 3):
                rise_ratio_second[k][t], w_divid[t], w_diff[t], ask[t], bid[t] = \
                    self.feature_extractor(trade_session[t][0], trade_session[t][1], w_list, rise_ratio_ab[k])

        data = {'ask': [[], [], []], 'bid': [[], [], []]}
        for k in data.keys():
            for it in range(0, 3):
                data[k][it] = np.array([label[k][it], list(range(trade_session[it][0], trade_session[it][1])),
                                        ask[it], bid[it]])
                for ic in range(0, len(rise_ratio_second[k][it])):
                    data[k][it] = np.append(data[k][it], [rise_ratio_second[k][it][ic]], axis=0)
                for ic in range(0, len(w_divid[it])):
                    data[k][it] = np.append(data[k][it], [w_divid[it][ic]], axis=0)
                for ic in range(0, len(w_diff[it])):
                    data[k][it] = np.append(data[k][it], [w_diff[it][ic]], axis=0)

                pd.DataFrame(data[k][it].T).to_csv(os.path.join(po, "Features_%s_S%d_%s" % (k, it, fn) + ".csv"), index=False, header=False)


if __name__ == "__main__":

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.split(proj_dir)[0], "data\\stock index future")

    t1 = time.clock()

    ftextractor = FeatureExtractor(proj_dir, data_dir, 600)

    for root, sub_dirs, files in os.walk(os.path.join(data_dir, "Data_by_Day")):
        for s_file in files:
            fn = s_file.split('.')[0]
            print(fn)
            ftextractor.output_data_set(fn, "Data_by_Day", "Features")
            # break

    print(u"总耗时 %.2f sec" % (time.clock()-t1))