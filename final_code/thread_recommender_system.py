#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:18:35 2020

@author: LMD
"""

import pandas as pd 
from preprocessing import preprocessing
from model_util import model_util
from result_analysis import result_analysis


def run_CLIR(
    dataset_name,
    experiment_setting,
    metric_type,
    load_model,
    nb_epoch,
    recommend_thread_to_users,
    add_user_id,
    add_time_dif
):
    # from numpy.random import seed
    # seed(1)
    # from tensorflow import set_random_seed
    # set_random_seed(2)
    
    #file paths
    model_name = 'CLIR_metrics_' + metric_type
    data_folder_path = "./thread_recommendation/data/{}/".format(dataset_name)
    data_file = \
        data_folder_path + 'dataset_{}.csv'.format(experiment_setting)
    token_source_path = data_folder_path + 'raw_data.csv'
    token_file_path = data_folder_path + 'tokenizer.pickle'
    user2reply_thread_path = data_folder_path + 'user_to_reply_thread.csv'
    thread2reply_user_path = data_folder_path + 'thread_to_replied_user.csv'
    result_path = \
        data_folder_path + 'results/'
    result_model_path = result_path + 'CLIR/'
    result_folder_path = result_model_path + experiment_setting + '/'
    model_folder_path = \
        data_folder_path + "{}_model/".format(experiment_setting)
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
    dropout_rate = 0.25
    # nb_epoch = 5
    topic_dim = 20
    output_dim = 2
    pool_size = 2
    
    #properties
    token_source_columns = [
        'init_post', 
        'replies'
    ]
    thread_recommendation_content_columns = [
        'target_users_recent_posts', 
        'initial_post_to_recommend'
    ]
    thread_recommendation_topic_columns = [
        'target_users_topic_vector',
        'candidate_thread_topic_vector'
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
    
    #generate input data
    target_users_recent_posts, \
    initial_post_to_recommend, \
    target_users_topic_vector, \
    candidate_thread_topic_vector, \
    Y, time, \
    train_or_test = pp.get_recommendation_input(
        data, 
        tokenizer, 
        thread_recommendation_content_columns,
        thread_recommendation_topic_columns,
        maxlen
    )
    
    #split data for training/testing
    post1_text_train, post1_text_test, \
    post2_text_train, post2_text_test, \
    post1_topic_train, post1_topic_test, \
    post2_topic_train, post2_topic_test, \
    Y_train_input, Y_test, \
    user_ids_train, user_ids_test, \
    user_thread_train, user_thread_test, \
    time_dif_train, time_dif_test = \
    pp.split_recommendation_data(
        data,
        target_users_recent_posts, 
        initial_post_to_recommend, 
        target_users_topic_vector, 
        candidate_thread_topic_vector, 
        Y, time, 
        train_or_test
    )
    
    #model training
    if not load_model:
        model = mu.get_CLIR(
            maxlen,
            topic_dim,
            dropout_rate,
            dense_size,
            output_dim,
            metric_type,
            max_features,
            embed_dim,
            filter_num,
            kernel_size,
            pool_size,
            add_user_id,
            add_time_dif
        )
        
        input_data = [
            post1_text_train, 
            post2_text_train,
            post1_topic_train,
            post2_topic_train
        ]
        
        if add_user_id:
            input_data += [user_ids_train]
        if add_time_dif:
            input_data += [time_dif_train]
        
        model.fit(
            input_data, 
            Y_train_input, 
            epochs=nb_epoch, 
            batch_size=batch_size, 
            verbose = 2
        )
        
        #save trained model
        pa.save_model(
            model, 
            model_path
        )
        
    else:
        model = pp.load_model(model_path)
    
    #predict on testing dataset
    input_data = [
        post1_text_test, 
        post2_text_test,
        post1_topic_test,
        post2_topic_test
    ]
    
    if add_user_id:
        input_data += [user_ids_test]
    if add_time_dif:
        input_data += [time_dif_test]
    
    y_p = model.predict(input_data, batch_size=batch_size, verbose=2)
    
    if recommend_thread_to_users:
        user2pred_thread = pa.gen_user_thread_rank_list(user_thread_test, y_p)
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
        


