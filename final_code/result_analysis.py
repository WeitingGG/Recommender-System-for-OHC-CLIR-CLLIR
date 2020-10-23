#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:42:49 2020

@author: LMD
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
)
import numpy as np 
import collections
from math import log
import csv
from preprocessing import preprocessing

class result_analysis(object):
    def __init__(self):
        pass
    
    def evaluate_results(self, y_p, test_Y, res, threshold):
        Y_pred = [1 if v[0] > threshold else 0 for v in y_p]       
        
        precision = precision_score(test_Y, Y_pred)
        recall = recall_score(test_Y, Y_pred)
        f1 = 2 * precision * recall / (precision + recall)
        acc = accuracy_score(test_Y, Y_pred)
        fpr, tpr, thresholds = roc_curve(test_Y, res, pos_label=1)
        auc = np.trapz(tpr, fpr)
        
        print('precision: %f' % precision)
        print('recall: %f' % recall)
        print('f1: %f' % f1)
        print('acc: %f' % acc)
        print('auc: %f' % auc)
        print('%.3f' % precision)
        print('%.3f' % recall)
        print('%.3f' % f1)
        print('%.3f' % acc)
        print('%.3f' % auc)
        return fpr, tpr
    
    def save_model(self, model, model_path):
        if model_path[-1] != '/':
            model_path += '/'
        model_json = model.to_json()
        with open(model_path+"model.json","w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_path+"model.h5")
        print("Saved model to disk")
    
    
    def save_results(self, results, result_path):
        with open(result_path + 'metrics_results.txt', 'w') as f:
            for item in results:
                f.write("%s\n" % item)
        print('Result saved successfully!')
    
    
    def gen_user_thread_rank_list(self, user_thread_test, y_p):
        user2pred_thread = collections.defaultdict(list)
        for i in range(len(user_thread_test)):
            user2pred_thread[user_thread_test[i][0]].append(
                [y_p[i][0], user_thread_test[i][1]]
            )
        for key, val in user2pred_thread.items():
            user2pred_thread[key] = sorted(val, reverse=True)
        return user2pred_thread
    
    
    def gen_thread_user_rank_list(self, user_thread_test, y_p):
        thread2pred_user = collections.defaultdict(list)
        for i in range(len(user_thread_test)):
            thread2pred_user[user_thread_test[i][1]].append(
                [y_p[i][1], user_thread_test[i][0]]
            )
        for key, val in thread2pred_user.items():
            thread2pred_user[key] = sorted(val, reverse=True)
        return thread2pred_user
    
    
    def gen_thread_user_ranking_metrics(
            self,
            thread2pred_user,
            thread2user_st,
            result_folder_path
    ):
        recall5 = []
        recall10 = []
        recall30 = []
        recall50 = []
        recall100 = []
        
        ndcg5 = []
        ndcg10 = []
        ndcg30 = []
        ndcg50 = []
        ndcg100 = []
        
        mrr = []
        
        ranked_thread = set()
        thread2metrics = []
        for thread_id, score2user_list in thread2pred_user.items():
            cnt5 = cnt10 = cnt30 = cnt50 = cnt100 = 0
            z5 = z10 = z30 = z50 = z100 = 0
            dcg5 = dcg10 = dcg30 = dcg50 = dcg100 = 0
            cnt = 0
            idx = 0
            while idx < len(score2user_list):
                user_id = str(score2user_list[idx][1])
                if user_id in thread2user_st[str(thread_id)]:
                    break
                idx += 1
            if idx != len(score2user_list):
                mrr.append(1 / (idx + 1))
            for i in range(100):
                if i < len(score2user_list):
                    score, user_id = score2user_list[i]
                    if user_id in thread2user_st[str(thread_id)]:
                        cnt += 1
                        if i < 5:
                            cnt5 += 1
                            dcg5 += 1.0 / log(i + 2, 2)
                        if i < 10:
                            cnt10 += 1
                            dcg10 += 1.0 / log(i + 2, 2)
                        if i < 30:
                            cnt30 += 1
                            dcg30 += 1.0 / log(i + 2, 2)
                        if i < 50:
                            cnt50 += 1
                            dcg50 += 1.0 / log(i + 2, 2)
                        if i < 100:
                            cnt100 += 1
                            dcg100 += 1.0 / log(i + 2, 2)
            if cnt == 0:
                continue
            for i in range(cnt):
                if i < 5:
                    z5 += 1.0 / log(i + 2, 2)
                if i < 10:
                    z10 += 1.0 / log(i + 2, 2)
                if i < 30:
                    z30 += 1.0 / log(i + 2, 2)
                if i < 50:
                    z50 += 1.0 / log(i + 2, 2)
                if i < 100:
                    z100 += 1.0 / log(i + 2, 2)
            # cnt = len(thread2user_st[str(thread_id)])
            ranked_thread.add(thread_id)
            recall5.append(1.0 * cnt5 / cnt)
            recall10.append(1.0 * cnt10 / cnt)
            recall30.append(1.0 * cnt30 / cnt)
            recall50.append(1.0 * cnt50 / cnt)
            recall100.append(1.0 * cnt100 / cnt)
            ndcg5.append(dcg5 / z5)
            ndcg10.append(dcg10 / z10)
            ndcg30.append(dcg30 / z30)
            ndcg50.append(dcg50 / z50)
            ndcg100.append(dcg100 / z100)
            thread2metrics.append([
                thread_id,
                recall5[-1],
                recall10[-1],
                recall30[-1],
                recall50[-1],
                recall100[-1],
                ndcg5[-1],
                ndcg10[-1],
                ndcg30[-1],
                ndcg50[-1],
                ndcg100[-1],
                mrr[-1]
            ])
            
        print('recall@5: %f' % (sum(recall5) / len(recall5)))
        print('recall@10: %f' % (sum(recall10) / len(recall10)))
        print('recall@30: %f' % (sum(recall30) / len(recall30)))
        print('recall@50: %f' % (sum(recall50) / len(recall50)))
        print('recall@100: %f' % (sum(recall100) / len(recall100)))
        
        print('ndcg@5: %f' % (sum(ndcg5) / len(ndcg5)))
        print('ndcg@10: %f' % (sum(ndcg10) / len(ndcg10)))
        print('ndcg@30: %f' % (sum(ndcg30) / len(ndcg30)))
        print('ndcg@50: %f' % (sum(ndcg50) / len(ndcg50)))
        print('ndcg@100: %f' % (sum(ndcg100) / len(ndcg100)))
        
        print('%.3f' % (sum(recall5) / len(recall5)))
        print('%.3f' % (sum(recall10) / len(recall10)))
        print('%.3f' % (sum(recall30) / len(recall30)))
        print('%.3f' % (sum(recall50) / len(recall50)))
        print('%.3f' % (sum(recall100) / len(recall100)))
        
        print('%.3f' % (sum(ndcg5) / len(ndcg5)))
        print('%.3f' % (sum(ndcg10) / len(ndcg10)))
        print('%.3f' % (sum(ndcg30) / len(ndcg30)))
        print('%.3f' % (sum(ndcg50) / len(ndcg50)))
        print('%.3f' % (sum(ndcg100) / len(ndcg100)))

        print('%.3f' % (sum(mrr) / len(mrr)))
        print('mrr: %f' % (sum(mrr) / len(mrr)))
        
        results = [
            sum(recall5) / len(recall5),
            sum(recall10) / len(recall10),
            sum(recall30) / len(recall30),
            sum(recall50) / len(recall50),
            sum(recall100) / len(recall100),
            sum(ndcg5) / len(ndcg5),
            sum(ndcg10) / len(ndcg10),
            sum(ndcg30) / len(ndcg30),
            sum(ndcg50) / len(ndcg50),
            sum(ndcg100) / len(ndcg100),
            sum(mrr) / len(mrr)
        ]
        
        with open(
                result_folder_path + 'thread_to_metrics.csv', 
                mode='w'
        ) as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow([
                'thread_id', 
                'recall@5',
                'recall@10',
                'recall@30',
                'recall@50',
                'recall@100',
                'ndcg@5',
                'ndcg@10',
                'ndcg@30',
                'ndcg@50',
                'ndcg@100',
                'mrr'
            ])
            for v in thread2metrics:
                tuple_writer.writerow(v)
        print('Metrics saved')
        
        return ranked_thread, results
    
    
    def gen_user_thread_ranking_metrics(
            self, 
            user2pred_thread, 
            user2reply_thread
    ):
        recall5 = []
        recall10 = []
        recall30 = []
        recall50 = []
        recall100 = []
        
        ndcg5 = []
        ndcg10 = []
        ndcg30 = []
        ndcg50 = []
        ndcg100 = []
        
        ranked_users = set()
        
        for user_id, val in user2pred_thread.items():
            if user_id not in user2reply_thread:
                continue
            cnt5 = cnt10 = cnt30 = cnt50 = cnt100 = 0
            z5 = z10 = z30 = z50 = z100 = 0
            dcg5 = dcg10 = dcg30 = dcg50 = dcg100 = 0
            cnt = 0
            for i in range(len(val)):
                thread_id = str(val[i][1])
                if thread_id in user2reply_thread[user_id]:
                    cnt += 1
                    if i < 5:
                        cnt5 += 1
                        dcg5 += 1.0 / log(i + 2, 2)
                    if i < 10:
                        cnt10 += 1
                        dcg10 += 1.0 / log(i + 2, 2)
                    if i < 30:
                        cnt30 += 1
                        dcg30 += 1.0 / log(i + 2, 2)
                    if i < 50:
                        cnt50 += 1
                        dcg50 += 1.0 / log(i + 2, 2)
                    if i < 100:
                        cnt100 += 1
                        dcg100 += 1.0 / log(i + 2, 2)
                if i < 5:
                    z5 += 1.0 / log(i + 2, 2)
                if i < 10:
                    z10 += 1.0 / log(i + 2, 2)
                if i < 30:
                    z30 += 1.0 / log(i + 2, 2)
                if i < 50:
                    z50 += 1.0 / log(i + 2, 2)
                if i < 100:
                    z100 += 1.0 / log(i + 2, 2)
            if cnt == 0:#filter out those who have no replies
                continue
            ranked_users.add(user_id)
            recall5.append(1.0 * cnt5 / cnt)
            recall10.append(1.0 * cnt10 / cnt)
            recall30.append(1.0 * cnt30 / cnt)
            recall50.append(1.0 * cnt50 / cnt)
            recall100.append(1.0 * cnt100 / cnt)
            ndcg5.append(dcg5 / z5)
            ndcg10.append(dcg10 / z10)
            ndcg30.append(dcg30 / z30)
            ndcg50.append(dcg50 / z50)
            ndcg100.append(dcg100 / z100)
            
        print('recall@5: %f' % (sum(recall5) / len(recall5)))
        print('recall@10: %f' % (sum(recall10) / len(recall10)))
        print('recall@30: %f' % (sum(recall30) / len(recall30)))
        print('recall@50: %f' % (sum(recall50) / len(recall50)))
        print('recall@100: %f' % (sum(recall100) / len(recall100)))
        
        print('ndcg@5: %f' % (sum(ndcg5) / len(ndcg5)))
        print('ndcg@10: %f' % (sum(ndcg10) / len(ndcg10)))
        print('ndcg@30: %f' % (sum(ndcg30) / len(ndcg30)))
        print('ndcg@50: %f' % (sum(ndcg50) / len(ndcg50)))
        print('ndcg@100: %f' % (sum(ndcg100) / len(ndcg100)))
        
        print('%.3f' % (sum(recall5) / len(recall5)))
        print('%.3f' % (sum(recall10) / len(recall10)))
        print('%.3f' % (sum(recall30) / len(recall30)))
        print('%.3f' % (sum(recall50) / len(recall50)))
        print('%.3f' % (sum(recall100) / len(recall100)))
        
        print('%.3f' % (sum(ndcg5) / len(ndcg5)))
        print('%.3f' % (sum(ndcg10) / len(ndcg10)))
        print('%.3f' % (sum(ndcg30) / len(ndcg30)))
        print('%.3f' % (sum(ndcg50) / len(ndcg50)))
        print('%.3f' % (sum(ndcg100) / len(ndcg100)))
        
        mrr = []
        
        for user_id, val in user2pred_thread.items():
            if user_id not in user2reply_thread:
                continue
            idx = 0
            while idx < len(val):
                thread_id = str(val[idx][1])
                if thread_id in user2reply_thread[user_id]:
                    break
                idx += 1
            if idx == len(val):
                mrr.append(0)
            else:
                mrr.append(1 / (idx + 1))
        
        print('%.3f' % (sum(mrr) / len(mrr)))
        print('mrr: %f' % (sum(mrr) / len(mrr)))
        
        results = [
            sum(recall5) / len(recall5),
            sum(recall10) / len(recall10),
            sum(recall30) / len(recall30),
            sum(recall50) / len(recall50),
            sum(recall100) / len(recall100),
            sum(ndcg5) / len(ndcg5),
            sum(ndcg10) / len(ndcg10),
            sum(ndcg30) / len(ndcg30),
            sum(ndcg50) / len(ndcg50),
            sum(ndcg100) / len(ndcg100),
            sum(mrr) / len(mrr)
        ]
        
        return ranked_users, results
    
    
    def save_thread_user_recommendation_list(
            self,
            result_folder_path,
            metric_type,
            thread2pred_user,
            ranked_threads,
            user2reply_thread
    ):
        with open(
                result_folder_path + 'recommend_list_{}.csv'.format(metric_type), 
                mode='w'
        ) as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(['thread_id', 'recommend_list'])
            for thread_id, user_list in thread2pred_user.items():
                if thread_id not in ranked_threads:
                    continue
                tuple_writer.writerow(
                    [
                        thread_id, 
                        ','.join(
                            [
                                str(v[1]) + ' ' \
                                + str(
                                    v[1] in user2reply_thread and \
                                    str(thread_id) in user2reply_thread[v[1]]
                                )
                                for v in user_list
                            ]
                        )
                    ]
                )
        print('Recommendation list is saved to disk')
    
    
    def save_thread_user_recommendation_list_with_score(
            self,
            result_folder_path,
            metric_type,
            thread2pred_user,
            ranked_threads,
            user2reply_thread
    ):
        with open(
                result_folder_path + 'recommend_list_{}.csv'.format(metric_type), 
                mode='w'
        ) as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(['thread_id', 'recommend_list'])
            for thread_id, user_list in thread2pred_user.items():
                if thread_id not in ranked_threads:
                    continue
                tuple_writer.writerow(
                    [
                        thread_id, 
                        ','.join(
                            [
                                str(v[1]) + ' ' \
                                + str(
                                    v[1] in user2reply_thread and \
                                    str(thread_id) in user2reply_thread[v[1]]
                                ) + ' ' + str(v[0])
                                for v in user_list
                            ]
                        )
                    ]
                )
        print('Recommendation list is saved to disk')
    
    def save_user_thread_recommendation_list(
            self,
            result_folder_path,
            metric_type,
            user2pred_thread,
            ranked_users,
            user2reply_thread
    ):
        with open(
                result_folder_path + 'recommend_list_{}.csv'.format(metric_type), 
                mode='w'
        ) as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(['user_id', 'recommend_list'])
            for user_id, thread_list in user2pred_thread.items():
                if user_id not in ranked_users:
                    continue
                tuple_writer.writerow(
                    [
                        user_id, 
                        ','.join(
                            [
                                str(v[1]) + ' ' \
                                + str(str(v[1]) in user2reply_thread[user_id]) \
                                + ' ' + str(v[0])
                                for v in thread_list
                            ]
                        )
                    ]
                )
        print('Recommendation list is saved to disk')