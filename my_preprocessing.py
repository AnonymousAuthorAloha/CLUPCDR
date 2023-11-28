import pandas as pd
import gzip
import json
import tqdm
import random
import os
import entry

class DataPreprocessingReady():
    def __init__(self,
                 root,
                 src_tgt_pairs,
                 task,
                 ratio):
        self.root = root
        self.src = src_tgt_pairs[task]['src']
        self.tgt = src_tgt_pairs[task]['tgt']
        self.ratio = ratio

    def read_mid(self, field):
        path = self.root + 'mid/' + field + '.csv'
        re = pd.read_csv(path)
        return re

    def mapper(self, src, tgt):
        print('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
        print('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
        co_uid = set(src.uid) & set(tgt.uid)
        all_uid = set(src.uid) | set(tgt.uid)
        print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))
        uid_dict = dict(zip(all_uid, range(len(all_uid))))
        iid_dict_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
        iid_dict_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
        src.uid = src.uid.map(uid_dict)
        src.iid = src.iid.map(iid_dict_src)
        tgt.uid = tgt.uid.map(uid_dict)
        tgt.iid = tgt.iid.map(iid_dict_tgt)
        return src, tgt

    def get_history(self, data, uid_set):
        pos_seq_dict = {}
        for uid in tqdm.tqdm(uid_set):
            selected_data = data[(data.uid == uid) & (data.y > 3)].iid.values.tolist()
            pos_iid=selected_data.iid.values.tolist()
            pos_ratings = selected_data.y.values.tolist()
            pos_seq_dict[uid]=pos_iid,pos_ratings
        return pos_seq_dict


    def split(self, src, tgt):
        print('All iid: {}.'.format(len(set(src.iid) | set(tgt.iid))))
        src_users = set(src.uid.unique())
        tgt_users = set(tgt.uid.unique())
        co_users = src_users & tgt_users
        test_users = set(random.sample(co_users, round(self.ratio[1] * len(co_users))))
        train_src = src
        train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]            #这里是关键，我们的数据及其结果都在目标领域当中，我们只要把测试集中的数据删除掉，就能得到训练数据集合了
        test = tgt[tgt['uid'].isin(test_users)]

        pos_seq_dict_source = self.get_history(src, co_users)
        pos_seq_dict_target = self.get_history(tgt, co_users)

        train_using_source=src[src['uid_src'].isin(co_users - test_users)]
        train_using_source['pos_seq_and_ratting_src'] = train_using_source['uid'].map(pos_seq_dict_source)
        train_using_target = tgt[tgt['uid_tgt'].isin(co_users - test_users)]
        train_using_target['pos_seq_and_ratting_tgt'] = train_using_target['uid'].map(pos_seq_dict_target)

        train_used_source = train_using_source.groupby('uid').first().reset_index()[['uid_src', 'pos_seq_and_ratting_src']]
        train_used_target = train_using_target.groupby('uid').first().reset_index()[['uid_tgt', 'pos_seq_and_ratting_tgt']]

        # 将'pos_seq_src'列拆分为两个独立的列
        train_used_source[['pos_seq_src', 'pos_seq_src_ratting']] = train_used_source['pos_seq_and_ratting_src'].apply(pd.Series)
        train_used_target[['pos_seq_tat', 'pos_seq_tgt_ratting']] = train_used_target['pos_seq_and_ratting_tgt'].apply(pd.Series)

        # 删除原始的'pos_seq_src'和'pos_seq_tgt'列
        train_used_source = train_used_source.drop('pos_seq_and_ratting_src', axis=1)
        train_used_target = train_used_target.drop('pos_seq_and_ratting_tgt', axis=1)

        # 横向拼接
        train_data = pd.concat([train_used_source, train_used_target], axis=1) 

        test['pos_seq_and_ratting_src'] = test['uid'].map(pos_seq_dict_source)
        test['pos_seq_and_ratting_tgt'] = test['uid'].map(pos_seq_dict_target)

        test[['pos_seq_src', 'pos_seq_src_ratting']] = test['pos_seq_and_ratting_src'].apply(pd.Series)
        test[['pos_seq_tat', 'pos_seq_tgt_ratting']] = test['pos_seq_and_ratting_tgt'].apply(pd.Series)

        test = test.drop('pos_seq_and_ratting_src', axis=1)
        test = test.drop('pos_seq_and_ratting_tgt', axis=1)


        return train_src, train_tgt,train_data,test

    def save(self, train_src, train_tgt, train_data, test):
        output_root = self.root + 'my_ready_2/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
                      '/tgt_' + self.tgt + '_src_' + self.src
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        print(output_root)
        train_src.to_csv(output_root + '/train_src.csv', sep=',', header=None, index=False)
        train_tgt.to_csv(output_root + '/train_tgt.csv', sep=',', header=None, index=False)
        train_data.to_csv(output_root +  '/train_data.csv', sep=',', header=None, index=False)
        test.to_csv(output_root +  '/test.csv', sep=',', header=None, index=False)

    def main(self):
        src = self.read_mid(self.src)
        tgt = self.read_mid(self.tgt)
        src, tgt = self.mapper(src, tgt)
        train_src, train_tgt, train_data,test = self.split(src, tgt)
        self.save(train_src, train_tgt, train_data, test)


if __name__ == '__main__':
    config_path = 'config.json'
    args, config = entry.prepare(config_path)

    # for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
    #     DataPreprocessingMid(config['root'], dealing).main()
    for ratio in [[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]]:
        for task in ['1', '2', '3']:
            DataPreprocessingReady(config['root'], config['src_tgt_pairs'], task, ratio).main()