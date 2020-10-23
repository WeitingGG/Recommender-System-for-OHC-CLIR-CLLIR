#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 03:41:38 2020

@author: LMD
"""

from thread_recommender_system import run_CLIR
from thread_recommender_system_with_interest_shifting import run_CLLIR

if __name__ == '__main__':
    # dataset_name = 'csn_breast_cancer'
    # experiment_setting = 'cold_start_thread'
    metric_type = 'recall'
    load_model = True
    CLIR_epoch = 5
    CLLIR_epoch = 5
    recommend_thread_to_users = False
    add_user_id = False
    add_time_dif = True
    post_limit = 5
                
    for dataset_name in ['stack']:
        for experiment_setting in ['cold_start_thread']:
            if experiment_setting == 'temporal_setting':
                recommend_thread_to_users = True
                add_time_dif = False
            run_CLIR(
                dataset_name,
                experiment_setting,
                metric_type,
                load_model,
                CLIR_epoch,
                recommend_thread_to_users,
                add_user_id,
                add_time_dif
            )
            print(
                'CLIR', 
                dataset_name, 
                experiment_setting
            )
            # run_CLLIR(
            #     dataset_name,
            #     experiment_setting,
            #     metric_type,
            #     load_model,
            #     CLLIR_epoch,
            #     recommend_thread_to_users,
            #     add_user_id,
            #     add_time_dif,
            #     post_limit
            # )
            # print(
            #     'CLLIR', 
            #     dataset_name, 
            #     experiment_setting
            # )