#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:41:16 2018

@author: LMD
"""

from preprocessing import preprocessing
from topic_modeling.LDA_program.run_lda import LDA_v20161130
from thread_recommender_system import run_CLIR
from thread_recommender_system_with_interest_shifting import run_CLLIR

dataset_name = 'stack'
experiment_setting = 'cold_start_thread'

html_dir_path = \
    '/Users/LMD/Desktop/Decision_Making/data/Html/{}/'.format(dataset_name)
thread_json_path = '../stack/{}.json'.format(dataset_name)
filter_thread_json_path = \
    './thread_recommendation/data/{}_{}.json'.format(dataset_name, 'filtered')
data_folder_path = "../{}/".format(dataset_name)
decision_making_thread_id_list = data_folder_path + 'idresult.txt'
topic_vector_path = data_folder_path + "decision_doc_topic.txt"
id_topic_vector_path = data_folder_path + "id_topic_vector.csv"
raw_data_path = data_folder_path + 'raw_data.csv'
user2reply_thread_path = data_folder_path + 'user_to_reply_thread.csv'
thread2reply_user_path = data_folder_path + 'thread_to_replied_user.csv'

pp = preprocessing()
#lda parameters
topic_dim = 20
lda_it_num = 1500
n_top_word = 10

#train/test parameters
train_ratio = 0.5
post_limit = 5
train_sample_num = 70000
test_sample_num = 30000
user_limit = 200
train_thread_limit = 300
test_thread_limit = 200

#experiment setting parameters
cold_start_user = 'user' in experiment_setting
cold_start_thread = 'thread' in experiment_setting
data_name = 'dataset_' + experiment_setting
use_all_threads = True
random_sample_threads = True
random_sample_users = False
use_ratio = True
time_oblivious = False

train_dates = set(
    # [
    #     '{}-{}-{}'.format(year, month, day) 
    #     for year in [2010] 
    #         for month in range(12, 13)
    #             for day in range(1, 32)
    # ]
    # [
    #     '{}-{}-{}'.format(year, month, day) 
    #     for year in [2011] 
    #         for month in [1,2,3]
    #             for day in range(1, 32)
    # ] 
    # + 
    [
        '{}-{}-{}'.format(year, month, day) 
        for year in [2010] 
            for month in [12]
                for day in range(1, 32)
    ]
)
test_dates = set(
    # [
    #     '{}-{}-{}'.format(year, month, day) 
    #     for year in [2011] 
    #         for month in range(1, 2)
    #             for day in range(1, 32)
    # ]
    [
        '{}-{}-{}'.format(year, month, day) 
        for year in [2011] 
            for month in [1]
                for day in range(1, 32)
    ]
)

# pp.get_rank_list(data_folder_path)

# pp.get_thread_number_distribution_per_user(
#     data_folder_path + 'dataset_{}.csv'.format(experiment_setting)
# )
# pp.get_statistic_of_thread_recommendation(
#     data_folder_path + 'dataset_{}.csv'.format(experiment_setting)
# )
# """
# Check the number of threads in each year
# """
# pp.get_thread_number_by_year(thread_json_path)
# pp.create_dirs([data_folder_path])
# # """
# # Step0: extract the thread from html files and save into json file
# # """
# pp.get_thread_from_html(html_dir_path, thread_json_path)
# pp.gen_filtered_threads(thread_json_path, filter_thread_json_path)
# """
# Step1: generate raw data csv from json file
# """
# pp.gen_csv_from_json(thread_json_path, raw_data_path)
# """
# Step2: get initial posts from json file
# """
# pp.gen_initial_posts(thread_json_path, data_folder_path, use_all_threads)
# """
# Step3: run lda on the 
# """
# _lda = LDA_v20161130(topics=topic_dim)
# _lda.setStopWords(_lda.stop)
# _lda.loadCorpusFromFile(data_folder_path + 'all_posts_intial.txt')
# _lda.fitModel(n_iter=lda_it_num)
# _lda.printTopic_Word(n_top_word=n_top_word)
# #_lda.printDoc_Topic()
# _lda.saveVocabulary(data_folder_path + 'decsion_vocab.txt')
# _lda.saveTopic_Words(data_folder_path + 'decision_topic_word.txt')
# _lda.saveDoc_Topic(topic_vector_path)
# """
# Step4: generate id to topic mapping file
# """
# pp.gen_id_to_topic(
#     decision_making_thread_id_list, 
#     topic_vector_path, 
#     id_topic_vector_path
# )
"""
Step5: generate data for thread recommendation system
"""
# pp.get_dataset_for_CLIR(
#     thread_json_path, 
#     decision_making_thread_id_list, 
#     data_folder_path, 
#     id_topic_vector_path, 
#     train_ratio,
#     train_sample_num,
#     test_sample_num,
#     cold_start_user,
#     cold_start_thread,
#     data_name,
#     post_limit,
#     user_limit,
#     train_thread_limit,
#     test_thread_limit,
#     random_sample_threads,
#     random_sample_users,
#     use_ratio,
#     time_oblivious
# )
pp.get_dataset_for_CLIR_based_on_time(
    thread_json_path, 
    decision_making_thread_id_list, 
    data_folder_path, 
    id_topic_vector_path, 
    data_name,
    post_limit,
    train_dates,
    test_dates
)
# pp.get_dataset_for_CLIR_with_temporal_setting(
#     thread_json_path, 
#     decision_making_thread_id_list, 
#     data_folder_path, 
#     id_topic_vector_path, 
#     data_name,
#     post_limit,
#     user_limit
# )
"""
Step6: generate data for thread recommendation system with time
"""
pp.get_dataset_for_CLLIR(
    thread_json_path, 
    decision_making_thread_id_list, 
    data_folder_path, 
    id_topic_vector_path, 
    train_ratio, 
    post_limit,
    data_name
)
# """
# Step7: generate data for train/test percentage testing
# """
# percentage = 0.2
# pp.split_data_with_x_percent_train(
#     data_folder_path, 
#     data_name, 
#     percentage,
#     ''
# )
# pp.split_data_with_x_percent_train(
#     data_folder_path, 
#     data_name, 
#     percentage,
#     '_with_time'
# )


