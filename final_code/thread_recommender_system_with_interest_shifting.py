#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:30:07 2020

@author: LMD
"""

import pandas as pd 
from preprocessing import preprocessing
from model_util import model_util
from result_analysis import result_analysis
    

def run_CLLIR(
    dataset_name,
    experiment_setting,
    metric_type,
    load_model,
    nb_epoch,
    recommend_thread_to_users,
    add_user_id,
    add_time_dif,
    post_limit
):
    model_name = 'CLLIR_metrics_' + metric_type
    data_folder_path = "./thread_recommendation/data/{}/".format(dataset_name)
    data_file = \
        data_folder_path + 'dataset_{}_with_time.csv'.format(experiment_setting)
    token_source_path = data_folder_path + 'raw_data.csv'
    token_file_path = data_folder_path + 'tokenizer.pickle'
    user2reply_thread_path = data_folder_path + 'user_to_reply_thread.csv'
    thread2reply_user_path = data_folder_path + 'thread_to_replied_user.csv'
    result_path = \
        data_folder_path + 'results/'
    result_model_path = result_path + 'CLLIR/'
    result_folder_path = result_model_path + experiment_setting + '/'
    model_folder_path = data_folder_path + "{}_model/".format(experiment_setting)
    model_path = model_folder_path + model_name + '/'
        
    #parameters
    max_features = 20000
    maxlen = 300
    batch_size = 32
    embed_dim = 128
    kernel_size = 5
    batch_size = 32
    filter_num = 32
    dense_size = 32
    lstm_out = 8#32
    dropout_rate = 0.25
    # nb_epoch = 5
    topic_dim = 20
    time_dim = 1
    dropout_U=0.2
    dropout_W=0.2
    # post_limit = 5
    pool_size = 2
    
    #properties
    token_source_columns = [
        'init_post', 
        'replies'
    ]
    
    #load helpers
    pp = preprocessing()
    mu = model_util()
    pa = result_analysis()
    
    #create directory
    pp.create_dirs([
         model_folder_path, 
         model_path, 
         result_path,
         result_model_path,
         result_folder_path
    ])
    
    #preprocessing
    data = pd.read_csv(data_file)
    print('CSV file read successfully!')
    user2reply_thread = pp.load_user2reply_thread(user2reply_thread_path)
    
    #generate or load token
    tokenizer = pp.get_token(
        token_source_path, 
        token_file_path, 
        token_source_columns,
        max_features
    )
    
    target_users_recent_posts, initial_post_to_recommend, \
    target_users_topic_vector, candidate_thread_topic_vector, \
    time, Y, train_or_test = pp.get_recommendation_time_input(
        data, 
        tokenizer,
        maxlen,
        post_limit
    )
    
    post1_text_train, post1_text_test, \
    post2_text_train, post2_text_test, \
    post1_topic_train, post1_topic_test, \
    post2_topic_train, post2_topic_test, \
    Y_train_input, Y_test, \
    user_thread_train, user_thread_test, \
    post_topic_dif_train, post_topic_dif_test, \
    train_time, test_time, \
    user_ids_train, user_ids_test, \
    time_dif_train, time_dif_test = \
    pp.split_recommendation_time_data(
        data,
        post_limit,
        target_users_recent_posts, 
        initial_post_to_recommend,
        target_users_topic_vector, 
        candidate_thread_topic_vector,
        Y, 
        time, 
        train_or_test
    )
    
    if not load_model:
        #model training
        model = mu.get_CLLIR(
            maxlen,
            post_limit,
            topic_dim,
            time_dim,
            max_features,
            embed_dim,
            filter_num,
            kernel_size,
            pool_size,
            dense_size,
            dropout_rate,
            lstm_out,
            dropout_U,
            dropout_W,
            metric_type,
            add_user_id,
            add_time_dif
        )
        
        input_data = post1_text_train \
        + [post2_text_train] \
        + post1_topic_train \
        + [post2_topic_train] \
        + train_time 
        
        if add_user_id:
            input_data += [user_ids_train]
        if add_time_dif:
            input_data += [time_dif_train]
        
        model.fit(
            input_data, 
            Y_train_input, 
            epochs=nb_epoch, 
            batch_size=batch_size, 
            verbose=2
        )
        
        #save trained model
        pa.save_model(
            model, 
            model_path
        )
    else:
        model = pp.load_model(model_path)
    
    #predict on test dataset
    input_data = post1_text_test \
    + [post2_text_test] \
    + post1_topic_test \
    + [post2_topic_test] \
    + test_time 
    
    if add_user_id:
        input_data += [user_ids_test]
    if add_time_dif:
        input_data += [time_dif_test]
    
    y_p = model.predict(input_data, batch_size=batch_size, verbose=2)
    
    if recommend_thread_to_users:
        user2pred_thread = pa.gen_thread_user_rank_list(user_thread_test, y_p)
        ranked_users, results = pa.gen_user_thread_ranking_metrics(
            user2pred_thread, user2reply_thread
        )
        pa.save_results(results, result_folder_path)
        pa.save_user_thread_recommendation_list(
            result_folder_path,
            metric_type,
            user2pred_thread,
            ranked_users,
            user2reply_thread
        )
    else:
        thread2pred_user = pa.gen_thread_user_rank_list(user_thread_test, y_p)
        thread2user_st = pp.load_thread2user_list(thread2reply_user_path)
        ranked_threads, results = pa.gen_thread_user_ranking_metrics(
            thread2pred_user, thread2user_st, result_folder_path
        )
        pa.save_results(results, result_folder_path)
        pa.save_thread_user_recommendation_list(
            result_folder_path,
            metric_type,
            thread2pred_user,
            ranked_threads,
            user2reply_thread
        )