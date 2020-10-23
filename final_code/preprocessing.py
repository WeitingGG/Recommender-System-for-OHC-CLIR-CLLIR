#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 11:59:32 2018

@author: LMD
"""

import json
import nltk.data
import os, sys
import numpy as np
import pandas as pd 
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
import re
import csv
import random
from dateutil import parser
import datetime
import collections
import bisect
from os import path, mkdir
import pickle
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize, pos_tag
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from math import log
# import zlib 

class preprocessing(object):
    def __init__(self):
        self.attributes = [
            'user_id', 
            'thread_id', 
            'target_users_recent_posts', 
            'initial_post_to_recommend', 
            'target_users_topic_vector', 
            'candidate_thread_topic_vector', 
            'time', 
            'train_or_test',
            'label',
            'time_dif'
        ]
        self.excel_size_limit = 30000
        self.least_reply_number = 5
        self.question_word_set = set(
            [
                'who', 
                'what', 
                'when', 
                'where', 
                'why', 
                'how', 
                'is', 
                'can', 
                'does', 
                'do', 
                'have', 
                'has', 
                'did', 
                'any'
            ]
        )
        self.train_name = 'train'
        self.test_name = 'test'
        
    
    def block_print(self):
        sys.stdout = open(os.devnull, 'w')
    
    
    def enable_print(self):
        sys.stdout = sys.__stdout__
    
    
    def get_word2vec_model(self, data_path, output_path):
        data = self.load_data(data_path)
        sentences = []
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for i in range(len(data)):
            for j in range(len(data[i]["posts"])):
                for s in tokenizer.tokenize(data[i]["posts"][j]["post"]):
                    words = []
                    for w in nltk.word_tokenize(s):
                        w = w.lower()
                        if w[0] >= 'a' and w[0] <= 'z':
                            words.append(w)
                    sentences.append(words)
        #print(len(sentences))
        model = Word2Vec(sentences, workers=8, min_count=20)
        print(model)
        model.save(output_path)
    
    
    def get_thread_from_html(self, input_path, output_path):
        file_names = os.listdir(input_path)
        threads = []
        user_st = set()
        for file_name in file_names:
            if file_name[0] < '0' or file_name[0] > '9':
                continue
            fp = open(input_path + file_name, "r")
            soup = BeautifulSoup(fp, 'html.parser')
            names = []
            for name in soup.find_all('span', attrs={'class':'username'}):
                names.append(name.text)
            posts = []
            for post in soup.find_all(
                    'div', attrs={'class':'field-item even'}
            ):
                posts.append(post.text)
            times = []
            for time in soup.find_all('div', attrs={'class':'submitted'}):
                s = str(parser.parse(time.text))
                times.append(s)
            for time in soup.find_all('span', attrs={'class':'submitted'}):
                s = str(parser.parse(time.text))
                times.append(s)
            fp.close()
            if len(names) != len(posts) or len(names) != len(times):
                print(file_name)
                continue
            if len(names) == len(posts) \
                and len(names) >= 3 \
                and len(names) <= 100:
                thread = []
                for i in range(len(names)):
                    p = {}
                    p["name"] = names[i]
                    user_st.add(names[i])
                    p["post"] = posts[i]
                    p["time"] = times[i]
                    thread.append(p)
                t = {}
                t["id"] = file_name.split(".")[0]
                t["posts"] = thread
                threads.append(t)
        print('{} threads'.format(len(threads)))
        print('{} users'.format(len(user_st)))
        with open(output_path, "w") as write_file:
            json.dump(threads, write_file)
    
    
    def gen_filtered_threads(self, json_data_path, filter_data_path):
        data = self.load_data(json_data_path)
        filtered_data = []
        for i in range(len(data)):
            if 3 <= len(data[i]['posts']) <= 100:
                filtered_data.append(data[i])
        with open(filter_data_path, "w") as write_file:
            json.dump(filtered_data, write_file)
        print('Filtered thread saved successfully!')
    
    def get_post_num(self, data_path):
        data = self.load_data(data_path)
        num = 0
        for i in range(len(data)):
            num += len(data[i]["posts"])
        return len(data), num
        
        
    def get_sentence_num(self, data_path):
        data = self.load_data(data_path)
        num = 0
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for i in range(len(data)):
            for j in range(len(data[i]["posts"])):
                num += len(tokenizer.tokenize(data[i]["posts"][j]["post"]))
        print(num)


    def get_statistic(self, data_path, decision_making_thread_id_list):
        data = self.load_data(data_path)
        decision_making_threads = set(
            self.get_decision_making_thread_id_list(
                decision_making_thread_id_list
            )
        )
        user_num = collections.defaultdict(int)
        user_post_num = {}
        
        for i in range(len(data)):
            if data[i]['id'] not in decision_making_threads:
                continue
            cnt = 0
            author_name = data[i]["posts"][0]["name"]
            user_st = set()
            for j in range(len(data[i]["posts"])):
                user_name = data[i]["posts"][j]["name"]
                if user_name == 'Anonymous user (not verified)' \
                    or user_name == author_name:
                        continue
                user_st.add(user_name)
                if (data[i]["posts"][j]["name"] in user_post_num):
                    user_post_num[data[i]["posts"][j]["name"]] += 1
                else:
                    user_post_num[data[i]["posts"][j]["name"]] = 1
            user_num[len(user_st)] += 1
        user_post_num = list(user_post_num.values())
        user_post_num.sort()
        # print(user_num[-1])#7478
        # print(user_num[int(len(user_num) * 5.0 / 10.0)])#2
        # print(user_num[int(len(user_num) * 55 / 100)])#2
        # print(user_num[int(len(user_num) * 6.0 / 10.0)])#3
        # print(user_num[int(len(user_num) * 65 / 100)])#3
        # print(user_num[int(len(user_num) * 7.0 / 10.0)])#4
        # print(user_num[int(len(user_num) * 75 / 100)])#6
        # print(user_num[int(len(user_num) * 8.0 / 10.0)])#8
        # print(user_num[int(len(user_num) * 85 / 100)])#14
        # print(user_num[int(len(user_num) * 9.0 / 10.0)])#29
        # print(user_num[int(len(user_num) * 99 / 100)])#801 
        
        user_num = [(key, val) for key, val in user_num.items()]
        user_num.sort()
        tot = 0
        for user_cnt, cnt in user_num:
            tot += cnt
            print(user_cnt, tot)
        
    
    def get_user_and_thread(self, data_path, output_path):
        data = self.load_data(data_path)
        users = set([])
        threads = set([])
        for i in range(len(data)):
            threads.add(data[i]["id"])
            for j in range(len(data[i]["posts"])):
                users.add(data[i]["posts"][j]["name"])
        print("user number: ", len(users))
        print("thread number: ", len(threads))
        with open(output_path + "users", "w") as write_file:
            json.dump(list(users), write_file)
        with open(output_path + "threads", "w") as write_file:
            json.dump(list(threads), write_file)
         
            
    def load_data(self, data_path):
        with open(data_path, "r") as read_file:
            data = json.load(read_file)
        return data
    
    
    def load_users(self, users_path):
        with open(users_path, "r") as read_file:
            users = json.load(read_file)
        print(len(users))
        return users
    
    
    def load_threads(self, threads_path):
        with open(threads_path, "r") as read_file:
            threads = json.load(read_file)  
        return threads
        
        
    def load_weight_for_embedding(self, weight_path):    
        weight = np.loadtxt(weight_path, delimiter=' ', dtype=float)
        return weight                 
    
    
    def comment_map(self, data_path, thread_id):
        file_name = data_path + thread_id + ".html"
        fp = open(file_name, "r")
        soup = BeautifulSoup(fp, 'html.parser')
        names = [
            name.text 
            for name in soup.find_all('span', attrs={'class':'username'})
        ]
        posts = [
            post.text 
            for post in soup.find_all(
                    'div', attrs={'class':'field-item even'}
            )
        ]
        f = open(file_name, "r")
        node_with_page = str(f.read().encode('utf-8'))
        comment_levels = {}
        comments = []
        reply_map = {0: []}
        comment_level = {}
        comments_with_tags = re.findall(
            r'<a id=\"comment-[0-9]+\"></a>', node_with_page
        )
        this_page_comments = [re.search(r'[0-9]+', comment).group() 
            for comment in comments_with_tags]
        id2name = dict(zip(['0'] + this_page_comments,names))
        id2post = dict(zip(['0'] + this_page_comments,posts))
        comments.extend(this_page_comments)
        for comment in this_page_comments:
            reply_beginning = re.search(
                '</table>(?:(?!</table).)*?' + comment, node_with_page
            ).group()
            marker_with_id = re.sub('</table>', '', reply_beginning)
            level_indication = re.sub('<a id="' + comment, '', marker_with_id)
            comment_level[comment] = level_indication
        for comment in comment_level:
            unindent = re.findall('</div>', comment_level[comment])
            if re.findall('<div class="indented">', comment_level[comment]):
                comment_level[comment] = 1
            elif unindent:
                comment_level[comment] = -len(unindent)
            else:
                comment_level[comment] = 0
            reply_map[comment] = []
        for i in range(len(this_page_comments) - 1):
            comment_level[this_page_comments[i + 1]] += \
                comment_level[this_page_comments[i]]
            if comment_level[this_page_comments[i]] == 0:
                reply_map[0].append(this_page_comments[i])
        comment_levels.update(comment_level)
        for comment in comment_levels:
            before = comments[:comments.index(comment)]
            prev = comment_levels[comment] - 1
            for i in range(1, len(before) + 1):
                replying_to = before[-i]
                if comment_levels[replying_to] == prev:
                    reply_map[replying_to].append(comment)
                    break
        reply_map = {str(key) : val for key, val in reply_map.items() if val}
        print(reply_map)
        records = []
        for reply_id in reply_map.get('0'):
            if reply_id in reply_map:
                for reply_reply_id in reply_map.get(reply_id):
                    if (id2name[reply_reply_id] == id2name['0']):
                        records.append(
                            [
                                thread_id, 
                                id2post['0'], 
                                id2post[reply_id], 
                                id2post[reply_reply_id]
                            ]
                        )
        f.close()
        fp.close()
        return records   


    def get_tuples(
            self, 
            decision_making_thread_id_list, 
            data_path, 
            tuple_csv_file_name
    ):
        fp = open(decision_making_thread_id_list, "r")
        lines = fp.readlines()
        with open(tuple_csv_file_name, mode='w') as tuple_csv_file:
            tuple_writer = csv.writer(
                tuple_csv_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(
                [
                    'thread_id', 
                    'initial_post', 
                    'reply_post', 
                    'initial_author_reply'
                ]
            )
            for _thread_id in lines:
                thread_id = _thread_id[:-1]
                records = self.comment_map(data_path, thread_id)
                for record in records:
                    tuple_writer.writerow(record)
        fp.close() 
    
    
    def get_decision_making_thread_id_list(
            self, 
            decision_making_thread_id_list
    ):
        fp = open(decision_making_thread_id_list, "r")
        lines = fp.readlines()
        return [each[:-1] for each in lines]
        
    
    def get_relevant_pairs(self, data, posts, train_idxs, relevant_pair_path):
        train_data = []
        test_data = []
        train_Y = []
        test_Y = []
        for i in range(len(data)):
            A = posts[0][i]
            B = posts[1][i]
            C = posts[2][i]
            if int(data['label'][i]) == 1:
                if i in train_idxs:
                    train_data.append([A, B])
                    train_data.append([B, C])
                    train_data.append([A, C])
                    train_data.append([B, A])
                    train_data.append([C, B])
                    train_data.append([C, A])
                    train_Y += [1] * 6
                else:
                    test_data.append([A, B])
                    test_data.append([B, C])
                    test_data.append([A, C])
                    test_data.append([B, A])
                    test_data.append([C, B])
                    test_data.append([C, A])
                    test_Y += [1] * 6
            else:
                ab = int(data["AB"][i])
                bc = int(data["BC"][i])
                if ab != -1:
                    if i in train_idxs:
                        train_data.append([A, B])
                        train_data.append([B, A])
                        train_Y += [ab] * 2
                    else:
                        test_data.append([A, B])
                        test_data.append([B, A])
                        test_Y += [ab] * 2
                if bc != -1:
                    if i in train_idxs:
                        train_data.append([C, B])
                        train_data.append([B, C])
                        train_Y += [bc] * 2
                    else:
                        test_data.append([C, B])
                        test_data.append([B, C])
                        test_Y += [bc] * 2
                if ab == 1 and bc == 1:
                    if i in train_idxs:
                        train_data.append([A, C])
                        train_data.append([C, A])
                        train_Y += [1] * 2
                    else:
                        test_data.append([A, C])
                        test_data.append([C, A])
                        test_Y += [1] * 2
        #balance train data
        pos_train = sum(train_Y)
        neg_train = len(train_Y) - pos_train
        train_list = list(train_idxs)
        ps = [
            [train_list[i], train_list[j]] 
            for i in range(len(train_list))
                for j in range(i + 1, len(train_list))
        ]
        ps_idxs = random.sample(
            range(0, len(ps)), 
            pos_train * 2 - neg_train
        )
        for idx in ps_idxs:
            i, j = ps[idx]
            train_data.append([posts[0][i], posts[1][j]])
            train_Y.append(0)
        #balance test data
        pos_test = sum(test_Y)
        neg_test = len(test_Y) - pos_test
        test_list = [i for i in range(len(data)) if i not in train_idxs]
        ps = [
            [test_list[i], test_list[j]] 
            for i in range(len(test_list))
                for j in range(i + 1, len(test_list))
        ]
        ps_idxs = random.sample(
            range(0, len(ps)), 
            (pos_test - neg_test) // 2
        )
        for idx in ps_idxs:
            i, j = ps[idx]
            test_data.append([posts[0][i], posts[1][j]])
            test_Y.append(0)
            
        pos_train = sum(train_Y)
        neg_train = len(train_Y) - pos_train
        pos_test = sum(test_Y)
        neg_test = len(test_Y) - pos_test
        
        print('positive train: {}'.format(pos_train))
        print('negative train: {}'.format(neg_train))
        print('positive test: {}'.format(pos_test))
        print('negative test: {}'.format(neg_test))
        print('Generate relevance pairs successfully!')
        
        with open(relevant_pair_path, mode='w') as relevant_pair_file:
            relevance_writter = csv.writer(
                relevant_pair_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            relevance_writter.writerow(
                [
                    'post1', 
                    'post2', 
                    'label', 
                    'train_or_test'
                ]
            )
            for i in range(len(train_data)):
                relevance_writter.writerow(
                    train_data[i] + [train_Y[i]] + [self.train_name]
                )
            for i in range(len(test_data)):
                relevance_writter.writerow(
                    test_data[i] + [test_Y[i]] + [self.test_name]
                )
        print('Save the pairs into disk!')
        
        return train_data, test_data, \
                train_Y, test_Y
    
    
    def get_time_stamp(self, s):
        v = s.split()
        sign = ''
        for c in s:
            if not c.isdigit() and c != ':' and c != ' ':
                sign = c
                break
        s = s.replace(',', '')
        time_format = ''
        time_format='%Y-%m-%d %H:%M:%S'
        # if len(v[0].split(sign)[0]) == 4:
        #     time_format = '%Y{0}%m{0}%d'.format(sign)
        # else:
        #     time_format = '%m{0}%d{0}%Y'.format(sign)
        # colon = s.count(':')
        # if len(v) == 3:
        #     if colon == 2:
        #         time_format += '  %I:%M:%S %p'
        #     elif colon == 1:
        #         time_format += '  %I:%M %p'
        #     else:
        #         raise Exception(s, time_format)
        # else:
        #     if colon == 2:
        #         time_format += '  %H:%M:%S'
        #     elif colon == 1:
        #         time_format += '  %H:%M'
        #     else:
        #         raise Exception(s, time_format)
        time = datetime.datetime.strptime(s, time_format)
        return time
    
    
    def encode(self, strs):
        """Encodes a list of strings to a single string.
        
        :type strs: List[str]
        :rtype: str
        """
        ans = ''
        for s in strs:
            ans += str(len(s)) + ' '
            ans += s
        return ans


    def decode(self, s):
        """Decodes a single string to a list of strings.
        
        :type s: str
        :rtype: List[str]
        """
        ans = []
        idx, l = 0, len(s)
        while idx < l:
            val = 0
            while idx < l and s[idx].isdigit():
                val = val * 10 + int(s[idx])
                idx += 1
            idx += 1
            ans.append(s[idx:idx + val])
            idx += val
        return ans
    
    
    def get_thread_len(self, data_path, decision_making_thread_id_list):
        data = self.load_data(data_path)
        mp = collections.defaultdict(int)
        for t in data:
            thread_id = t['id']
            mp[thread_id] = len(t['posts'])
        return mp
    
    
    def collect_data(self, data, decision_making_threads):
        #initialize data
        thread_id_time = []
        user2reply_thread = collections.defaultdict(list)
        user2all_thread = collections.defaultdict(list)
        thread2initial_post = {}
        thread2initial_author = {}
        user_adj = collections.defaultdict(set)
        #collect data
        for t in data:
            if not t['posts']:
                continue
            initial_author_name = t['posts'][0]['name']
            thread_id = t['id']
            if initial_author_name == 'Anonymous user (not verified)' \
                or thread_id not in decision_making_threads:
                continue
            thread2initial_author[thread_id] = initial_author_name
            thread2initial_post[thread_id] = t['posts'][0]['post']
            thread_time = self.get_time_stamp(t['posts'][0]['time'][:-1])
            user2all_thread[initial_author_name].append(
                (thread_time, thread_id)
            )
            if [thread_id, thread_time] not in thread_id_time:
                thread_id_time.append([thread_id, thread_time])
            user_st = set()
            for i in range(1, len(t['posts'])):
                if t['posts'][i]['name'] == 'Anonymous user (not verified)' \
                    or t['posts'][i]['name'] == t['posts'][0]['name']:
                    continue
                reply_user_name = t['posts'][i]['name']
                user2all_thread[reply_user_name].append(
                    (thread_time, thread_id)
                )
                user2reply_thread[reply_user_name].append(
                    (thread_time, thread_id)
                )
                user_st.add(reply_user_name)
            user_list = list(user_st)
            for i in range(len(user_list)):
                for j in range(i + 1, len(user_list)):
                    user_adj[user_list[i]].add(user_list[j])
                    user_adj[user_list[j]].add(user_list[i])
                
        #sort by time stamp        
        thread_id_time.sort(key=lambda v : v[-1])
        user2thread_id = collections.defaultdict(set)
        for user, v in user2all_thread.items():
            user2all_thread[user] = sorted(v, key = lambda v : v[0])
            user2thread_id[user] = set([e[-1] for e in v])
        return thread_id_time, \
                user2reply_thread, \
                user2all_thread, \
                thread2initial_post, \
                thread2initial_author, \
                user_adj 
                
                
    def gen_old_new_threads(
            self, 
            thread_id_time,
            last_idx,
            train_thread_limit,
            test_thread_limit,
            random_sample_threads,
            time_oblivious
    ):
        #get train/test thread
        thread_num = len(thread_id_time)
        if time_oblivious:
            thread_idx = random.sample(
                range(0, thread_num - 1),
                train_thread_limit+test_thread_limit
            )
            train_thread_idx = set(
                random.sample(
                    thread_idx,
                    train_thread_limit
                )
            )
            old_thread = [
                thread_id_time[i]
                for i in train_thread_idx
            ]
            new_thread = [
                thread_id_time[i]
                for i in thread_idx 
                    if i not in train_thread_idx
            ]
        elif random_sample_threads:
            train_thread_idx = set(
                random.sample(
                    range(0, last_idx),
                    train_thread_limit
                )
            )
            old_thread = [
                thread_id_time[i]
                for i in train_thread_idx
            ]
            old_thread.sort(key=lambda v : v[-1])
            test_thread_idx = set(
                random.sample(
                    range(last_idx + 1, thread_num - 1),
                    test_thread_limit
                )
            )
            new_thread = [
                thread_id_time[i]
                for i in test_thread_idx
            ]
            new_thread.sort(key=lambda v : v[-1])
            print('Start time: {}'.format(old_thread[0][1]))
            print('Split time: {}'.format(old_thread[-1][1]))
            print('End time: {}'.format(new_thread[-1][1]))
        else:
            old_thread = [
                thread_id_time[i]
                for i in range(thread_num)
                    if i <= last_idx and i > last_idx - train_thread_limit
            ]
            new_thread = [
                thread_id_time[i]
                for i in range(thread_num)
                    if i > last_idx and i <= last_idx + test_thread_limit
            ]
        print('Old thread number: {}'.format(len(old_thread)))
        print('New thread number: {}'.format(len(new_thread)))
        return old_thread, new_thread
    
    
    def gen_all_users(
            self, 
            user2reply_thread
    ):
        #get train/test user
        all_users = [
            user_id 
            for user_id, v in user2reply_thread.items()
                if len(v) >= self.least_reply_number
        ]
        print('All user number: {}'.format(len(all_users)))
        return all_users
    
    
    def get_random_sample_pairs(
            self, 
            thread_list, 
            user_list,
            sample_num,
            user2all_thread,
            thread2initial_author
    ):
        pair_list = []
        user_st = set()
        thread_st = set()
        idx = 0
        for user_id in user_list:
            all_replied_thread_st = set(
                [v[-1] for v in user2all_thread[user_id]]
            )
            for thread_id, time in thread_list:
                assert(thread_id in thread2initial_author)
                if thread2initial_author[thread_id] != user_id:
                    label = \
                        1 if thread_id in all_replied_thread_st \
                        else 0
                    #use reservoir sampling:
                    #https://en.wikipedia.org/wiki/Reservoir_sampling
                    if len(pair_list) == sample_num:
                        rand_idx = random.randint(0, idx)
                        if rand_idx < sample_num:
                            pair_list[rand_idx] = (
                                thread_id,
                                user_id,
                                time,
                                label
                            )
                    else:
                        pair_list.append(
                            (
                                thread_id,
                                user_id,
                                time,
                                label
                            )
                        )
                    idx += 1
        for thread_id, user_id, time, label in pair_list:
            user_st.add(user_id)
            thread_st.add((thread_id, time))
        print('Pair list is generated successfully!')
        return pair_list, list(user_st), list(thread_st)
    
    
    def get_all_pairs(
            self, 
            thread_list, 
            user_list,
            user2all_thread,
            thread2initial_author,
            train_or_test
    ):
        pos_pair_list, neg_pair_list = [], []
        neg_pair_list
        user_st = set()
        thread_st = set()
        
        for user_id in user_list:
            all_replied_thread_st = set(
                [v[-1] for v in user2all_thread[user_id]]
            )
            for thread_id, time in thread_list:
                assert(thread_id in thread2initial_author)
                if thread2initial_author[thread_id] != user_id:
                    if thread_id in all_replied_thread_st:
                        pos_pair_list.append(
                            (
                                thread_id,
                                user_id,
                                time,
                                1
                            )
                        )
                    else:
                        neg_pair_list.append(
                            (
                                thread_id,
                                user_id,
                                time,
                                0
                            )
                        )
        if train_or_test == self.train_name:
            neg_pair_list = random.sample(
                neg_pair_list,
                int(len(pos_pair_list) * 1)
            )
        pair_list = pos_pair_list + neg_pair_list
        for thread_id, user_id, time, label in pair_list:
            user_st.add(user_id)
            thread_st.add((thread_id, time))
        print('Pair list is generated successfully!')
        print('There are {} pairs.'.format(len(pair_list)))
        return pair_list, list(user_st), list(thread_st)
    
    
    def get_involved_threads(
            self,
            user_list,
            user2reply_thread,
            candidate_threads,
            pair_num
    ):
        thread_st = set()
        for user_id in user_list:
           for v in user2reply_thread[user_id]:
               thread_st.add((v[1], v[0]))
        thread_list = list(thread_st)
        if len(user_list) * len(thread_list) > pair_num:
            thread_idx = set(
                random.sample(
                    range(0, len(thread_list)),
                    pair_num // len(user_list)
                )
            )
            thread_list = [
                thread_list[i] 
                for i in range(len(thread_list)) 
                    if i in thread_idx
            ]
        print('There are {} threads involved.'.format(len(thread_list)))
        return thread_list
    
    
    def gen_thread2users(self, user2reply_thread):
        thread2user_st = collections.defaultdict(set)
        for user_id, time2thread_list in user2reply_thread.items():
            for time, thread_id in time2thread_list:
                thread2user_st[thread_id].add(user_id)
        print('Generated thread to user set!')
        return thread2user_st
    
    
    def get_users_from_thread(
            self, 
            thread_id_time, 
            thread2user_st, 
            user2reply_thread
    ):
        user_st = set()
        for thread_id, time in thread_id_time:
            for user_id in thread2user_st[thread_id]:
                if len(user2reply_thread[user_id]) >= self.least_reply_number:
                    user_st.add(user_id)
        return list(user_st)
    
    
    def filter_test_users(self, train_users, test_users):
        train_user_st = set(train_users)
        # filtered_users = [
        #     test_user 
        #     for test_user in test_users if test_user not in train_user_st
        # ]
        test_users = [
            test_user 
            for test_user in test_users if test_user in train_user_st
        ]
        # print(len(filtered_users))
        # print(filtered_users)
        return test_users
    
    
    def save_thread2_user_list(self, user2reply_thread_path, data_path):
        user2reply_thread = self.load_user2reply_thread(user2reply_thread_path)
        thread2user_st = collections.defaultdict(set)
        for user_id, time2thread_list in user2reply_thread.items():
            for thread_id in time2thread_list:
                thread2user_st[thread_id].add(user_id)
        data = [
            [thread_id, ','.join([str(e) for e in user_list])]
            for thread_id, user_list in thread2user_st.items()
        ]
        self.save_csv(data, data_path, ['thread_id', 'user_list'])
        
        
    def load_thread2user_list(self, thread2user_list_path):
        data = pd.read_csv(thread2user_list_path)
        thread2user_st = collections.defaultdict(set)
        for i in range(len(data)):
            thread2user_st[str(data['thread_id'][i])] = \
                set(str(data['user_list'][i]).split(','))
        return thread2user_st
    
    
    def gen_train_test_sample_based_on_time(
            self,
            thread_id_time,
            user2reply_thread,
            user2all_thread,
            thread2initial_author,
            train_dates,
            test_dates
    ):
        #generate train/test samples
        all_thread_ids = set([thread_id for thread_id, time in thread_id_time])
        assert(len(all_thread_ids) == len(thread_id_time))
        train_samples, test_samples = [], []
        thread_id_date_time = [
            [
                thread_id, 
                '{}-{}-{}'.format(
                    str(time.year), 
                    str(time.month), 
                    str(time.day)
                ), 
                time
            ]
            for thread_id, time in thread_id_time
        ]
        train_thread_time = [
            [thread_id, time]
            for thread_id, date, time in thread_id_date_time
                if date in train_dates
        ]
        train_thread_time.sort(key=lambda v : v[-1])
        train_threads = set([
            thread_id
            for thread_id, time in train_thread_time
        ])
        test_thread_time = [
            [thread_id, time]
            for thread_id, date, time in thread_id_date_time
                if date in test_dates
        ]
        test_threads = set([
            thread_id 
            for thread_id, time in test_thread_time
        ])
        
        test_thread_authors = set()
        
        for thread_id in test_threads:
            test_thread_authors.add(thread2initial_author[thread_id])
            
        print('Test thread initial author number: {}'.format(
            len(test_thread_authors)
        ))
        
        train_users, test_users = [], []
        for user_id, time2thread_list in user2reply_thread.items():
            for time, thread_id in time2thread_list:
                if thread_id in train_threads:
                    train_users.append(user_id)
                    break
            for time, thread_id in time2thread_list:
                if thread_id in test_threads:
                    test_users.append(user_id)
                    break   
        
        #calculate average user activity number
        user2act_num = collections.defaultdict(int)
        for user_id in train_users:
            for thread_id in train_threads.union(test_threads):
                replied = thread2initial_author[thread_id] == user_id
                for time, reply_thread_id in user2reply_thread[user_id]:
                    if reply_thread_id == thread_id:
                        replied = True
                        break
                if replied:
                    user2act_num[user_id] += 1
        print('Activity number per user is {}'.format(
            sum(user2act_num.values()) / len(user2act_num)
        ))
        
        #check overlap user number
        test_users = self.filter_test_users(train_users, test_users)
        print('There are {} overlap users'.format(len(test_users)))
            
        train_samples, _, _ = \
        self.get_all_pairs(
            train_thread_time, 
            train_users,
            user2all_thread,
            thread2initial_author,
            self.train_name
        )
        test_samples, _, _ = \
        self.get_all_pairs(
            test_thread_time, 
            train_users,
            user2all_thread,
            thread2initial_author,
            self.test_name
        )
        
        #calculate sparsity of the data
        sparsity = \
            1.0 - sum([v[-1] for v in train_samples]) / len(train_samples)
        print('Sparsity is {}'.format(sparsity))
        
        print('Train user number: {}'.format(len(train_users)))
        print('Test user number: {}'.format(len(test_users)))
        print('Train thread number: {}'.format(len(train_threads)))
        print('Test thread number: {}'.format(len(test_threads)))
        print('Train pair number: {}'.format(len(train_samples)))
        print('Test pair number: {}'.format(len(test_samples)))
        return train_samples, test_samples
    
    
    def gen_train_test_sample_based_on_users(
            self,
            thread_id_time,
            user2reply_thread,
            user2all_thread,
            thread2initial_author,
            user_limit
    ):
        #generate train/test samples
        all_users = self.gen_all_users(user2reply_thread)
        all_users = random.sample(
            all_users,
            user_limit
        )
        train_threads, test_threads = set(), set()
        pos_train_samples, neg_train_samples, test_samples = [], [], []
        train_pos_pairs, test_pos_pairs = set(), set()
        thread2time = {}
        
        #generate positive samples
        for user_id in all_users:
            assert(user_id in user2reply_thread)
            time2thread_id = user2reply_thread[user_id]
            time2thread_id.sort()
            split_idx = len(time2thread_id) * 4 // 5
            for i in range(len(time2thread_id)):
                time, thread_id = time2thread_id[i]
                thread2time[thread_id] = time
                if i < split_idx:
                    train_pos_pairs.add((user_id, thread_id))
                    train_threads.add(thread_id)
                else:
                    test_pos_pairs.add((user_id, thread_id))
                    test_threads.add(thread_id)
        
        #generate negative samples
        for user_id in all_users:
            for thread_id in train_threads:
                if (user_id, thread_id) not in train_pos_pairs:
                    neg_train_samples.append((
                        thread_id,
                        user_id,
                        thread2time[thread_id],
                        0
                    ))
                else:
                    pos_train_samples.append((
                        thread_id,
                        user_id,
                        thread2time[thread_id],
                        1
                    ))
            for thread_id in test_threads:
                if (user_id, thread_id) not in test_pos_pairs:
                    test_samples.append((
                        thread_id,
                        user_id,
                        thread2time[thread_id],
                        0
                    ))
                else:
                    test_samples.append((
                        thread_id,
                        user_id,
                        thread2time[thread_id],
                        1
                    ))
        neg_train_samples = random.sample(
            neg_train_samples,
            int(len(pos_train_samples) * 1.0)
        )
        train_samples = pos_train_samples + neg_train_samples
        
        print('Selected user number: {}'.format(len(all_users)))
        print('Train thread number: {}'.format(len(train_threads)))
        print('Test thread number: {}'.format(len(test_threads)))
        print('Train pair number: {}'.format(len(train_samples)))
        print('Test pair number: {}'.format(len(test_samples)))
        return train_samples, test_samples
    
    
    def get_dataset_for_CLIR_with_temporal_setting(
            self,
            data_path, 
            decision_making_thread_id_list, 
            experiment_data_path, 
            id_topic_vector_path, 
            data_name,
            post_limit,
            user_limit
    ):
        #load data
        data = self.load_data(data_path)
        decision_making_threads = set(
            self.get_decision_making_thread_id_list(
                decision_making_thread_id_list
            )
        )
        id2topic = self.load_id_to_topic(id_topic_vector_path)
        print('Successfully load the data!')
        
        #collect data
        thread_id_time, \
        user2reply_thread, \
        user2all_thread, \
        thread2initial_post, \
        thread2initial_author, \
        user_adj = self.collect_data(data, decision_making_threads)
        
        #generate train/test samples
        train_samples, test_samples = \
        self.gen_train_test_sample_based_on_users(
            thread_id_time,
            user2reply_thread,
            user2all_thread,
            thread2initial_author,
            user_limit
        )
        
        #generate dataset
        dataset = self.gen_dataset(
            id2topic,
            train_samples,
            test_samples,
            user2all_thread,
            thread2initial_post,
            post_limit
        )
        
        #save dataset
        self.save_csv(
            dataset, 
            experiment_data_path + '{}.csv'.format(data_name), 
            self.attributes
        )
        
        #save user mapping to their replied threads
        self.save_csv(
            [
                [user_name, '|'.join([str(v[1]) for v in st])]
                for user_name, st in user2reply_thread.items()
                    if st
            ],
            experiment_data_path + 'user_to_reply_thread.csv',
            ['user_name', 'threads']
        )
        
        #save user mapping to their replied threads with time
        self.save_csv(
            [
                [user_name, '|'.join(
                    [str(v[1]) + ' ' + str(v[0]) for v in st]
                )]
                for user_name, st in user2reply_thread.items()
                    if st
            ],
            experiment_data_path + 'user_to_reply_thread_with_time.csv',
            ['user_name', 'threads']
        )
        
        #save thread id mapping to initial post
        self.save_csv(
            [
                [thread_id, initial_post]
                for thread_id, initial_post in thread2initial_post.items()
            ],
            experiment_data_path + 'thread_to_initial_post.csv',
            ['thread_id', 'initial_post']
        )
        
        #save user adjacency graph
        self.save_csv(
            [
                [user_name, '|'.join(list(st))]
                for user_name, st in user_adj.items()
            ],
            experiment_data_path + 'user_adj.csv',
            ['user_name', 'neighbors']
        )
        
        #save user thread id mapping to replied users
        self.save_thread2_user_list(
            experiment_data_path + 'user_to_reply_thread.csv',
            experiment_data_path + 'thread_to_replied_user.csv', 
        )
        
        #save thread_id to initial_author 
        self.save_csv(
            [
                [thread_id, initial_author]
                for thread_id, initial_author in thread2initial_author.items()
            ],
            experiment_data_path + 'thread_to_initial_author.csv',
            ['thread_id', 'user_name']
        )
    
    
    def get_dataset_for_CLIR_based_on_time(
            self, 
            data_path, 
            decision_making_thread_id_list, 
            experiment_data_path, 
            id_topic_vector_path, 
            data_name,
            post_limit,
            train_dates,
            test_dates
    ):
        #load data
        data = self.load_data(data_path)
        decision_making_threads = set(
            self.get_decision_making_thread_id_list(
                decision_making_thread_id_list
            )
        )
        id2topic = self.load_id_to_topic(id_topic_vector_path)
        print('Successfully load the data!')
        
        #collect data
        thread_id_time, \
        user2reply_thread, \
        user2all_thread, \
        thread2initial_post, \
        thread2initial_author, \
        user_adj = self.collect_data(data, decision_making_threads)
        
        #generate train/test samples
        train_samples, test_samples = \
        self.gen_train_test_sample_based_on_time(
            thread_id_time,
            user2reply_thread,
            user2all_thread,
            thread2initial_author,
            train_dates,
            test_dates
        )
        print(len(train_samples))
        print(len(test_samples))
        
        #generate dataset
        self.gen_dataset_based_on_time(
            id2topic,
            train_samples,
            test_samples,
            user2all_thread,
            thread2initial_post,
            post_limit,
            experiment_data_path,
            data_name
            )
        # dataset = self.gen_dataset(
        #     id2topic,
        #     train_samples,
        #     test_samples,
        #     user2all_thread,
        #     thread2initial_post,
        #     post_limit
        # )
        
        #save dataset
        # self.save_csv(
        #     dataset, 
        #     experiment_data_path + '{}.csv'.format(data_name), 
        #     self.attributes
        # )
        
        #save user mapping to their replied threads
        self.save_csv(
            [
                [user_name, '|'.join([str(v[1]) for v in st])]
                for user_name, st in user2reply_thread.items()
                    if st
            ],
            experiment_data_path + 'user_to_reply_thread.csv',
            ['user_name', 'threads']
        )
        
        #save user mapping to their replied threads with time
        self.save_csv(
            [
                [user_name, '|'.join(
                    [str(v[1]) + ' ' + str(v[0]) for v in st]
                )]
                for user_name, st in user2reply_thread.items()
                    if st
            ],
            experiment_data_path + 'user_to_reply_thread_with_time.csv',
            ['user_name', 'threads']
        )
        
        #save thread id mapping to initial post
        self.save_csv(
            [
                [thread_id, initial_post]
                for thread_id, initial_post in thread2initial_post.items()
            ],
            experiment_data_path + 'thread_to_initial_post.csv',
            ['thread_id', 'initial_post']
        )
        
        #save user adjacency graph
        self.save_csv(
            [
                [user_name, '|'.join(list(st))]
                for user_name, st in user_adj.items()
            ],
            experiment_data_path + 'user_adj.csv',
            ['user_name', 'neighbors']
        )
        
        #save user thread id mapping to replied users
        self.save_thread2_user_list(
            experiment_data_path + 'user_to_reply_thread.csv',
            experiment_data_path + 'thread_to_replied_user.csv', 
        )
        
        #save thread_id to initial_author 
        self.save_csv(
            [
                [thread_id, initial_author]
                for thread_id, initial_author in thread2initial_author.items()
            ],
            experiment_data_path + 'thread_to_initial_author.csv',
            ['thread_id', 'user_name']
        )
    
    
    def gen_train_test_sample(
            self,
            all_users,
            old_thread,
            new_thread,
            user2reply_thread,
            user2all_thread,
            thread2initial_author,
            train_sample_num,
            test_sample_num,
            time_th,
            cold_start_user,
            cold_start_thread,
            user_limit,
            random_sample_users
    ):
        #generate train/test samples
        if random_sample_users:
            train_samples, test_samples = [], []
            train_user_idx = set(
                random.sample(
                    range(0, len(all_users)),
                    min(user_limit, len(all_users))
                )
            )
            train_users = [
                all_users[i]
                for i in range(len(all_users))
                    if i in train_user_idx
            ]
        else:
            thread2user_st = self.gen_thread2users(user2reply_thread)
            train_users = self.get_users_from_thread(
                old_thread, 
                thread2user_st,
                user2reply_thread
            )
            # train_user_idx = set(
            #     random.sample(
            #         range(0, len(train_users)),
            #         min(user_limit, len(train_users))
            #     )
            # )
            # train_users = [
            #     all_users[i]
            #     for i in range(len(train_users))
            #         if i in train_user_idx
            # ]
            test_users = self.get_users_from_thread(
                new_thread, 
                thread2user_st,
                user2reply_thread
            )
            # test_users = self.filter_test_users(
            #     train_users, test_users, user_limit
            # )
        print('Train user number: {}'.format(len(train_users)))
        print('Test user number: {}'.format(len(test_users)))
        # train_user_idx = set(
        #     random.sample(
        #         range(0, len(all_users)),
        #         min(len(all_users), user_limit)
        #     )
        # )
        # train_users = [
        #     all_users[i]
        #     for i in range(len(all_users))
        #         if i in train_user_idx
        # ]
        # test_users = [
        #     all_users[i]
        #     for i in range(len(all_users))
        #         if i not in train_user_idx
        # ]
        if not cold_start_user and not cold_start_thread:
            train_samples, user_list, thread_list = \
            self.get_random_sample_pairs(
                new_thread, 
                train_users,
                train_sample_num,
                user2all_thread,
                thread2initial_author
            )
            test_samples, _, _ = \
            self.get_random_sample_pairs(
                thread_list, 
                user_list,
                test_sample_num,
                user2all_thread,
                thread2initial_author
            )
        elif cold_start_user and not cold_start_thread:
            pass
        elif not cold_start_user and cold_start_thread:
            train_samples, _, _ = \
            self.get_all_pairs(
                old_thread, 
                train_users,
                user2all_thread,
                thread2initial_author,
                self.train_name
            )
            test_samples, _, _ = \
            self.get_all_pairs(
                new_thread, 
                test_users,
                user2all_thread,
                thread2initial_author,
                self.test_name
            )
        else:
            pass
        print('Train pair number: {}'.format(len(train_samples)))
        print('Test pair number: {}'.format(len(test_samples)))
        return train_samples, test_samples
    
    
    def plus_topic_vector(self, v1, v2):
        for i in range(20):
            v1[i] += v2[i]
        return v1
    
    
    def gen_dataset(
            self,
            id2topic,
            train_samples,
            test_samples,
            user2all_thread,
            thread2initial_post,
            post_limit
    ):
        #initialization
        dataset = []
        train_pos = test_pos = train_neg = test_neg = 0
        #generate dataset
        cnt=0
        for samples, train_or_test in [
            [train_samples, self.train_name],
            [test_samples, self.test_name]
        ]:
            for thread_id, reply_user_name, time, label in samples:
                int_thread_id = int(thread_id)
                assert(int_thread_id in id2topic)
                cand_topic = ' '.join(
                    [str(val) for val in id2topic[int_thread_id]]
                )
                idx = bisect.bisect_left(
                    user2all_thread[reply_user_name], 
                    (time, '')
                )
                recent_post = []
                recent_post_topic = []
                num = 0
                time_dif = int(1e9)
                for i in range(idx - 1, -1, -1):
                    if user2all_thread[reply_user_name][i][1] == thread_id:
                        continue
                    recent_thread_time, recent_thread_id = \
                        user2all_thread[reply_user_name][i]
                    time_dif = min(
                        time_dif, 
                        int(
                            (
                                time - recent_thread_time
                            ).total_seconds() / 60
                        )
                    )
                    post_text = thread2initial_post[recent_thread_id]
                    if len(recent_post) + len(post_text) \
                        > self.excel_size_limit:
                        continue
                    int_recent_thread_id = int(recent_thread_id)
                    recent_post.append(post_text)
                    assert(int_recent_thread_id in id2topic)
                    recent_post_topic.append(id2topic[int_recent_thread_id])
                    num += 1
                
                while len(recent_post) < post_limit:
                    recent_post.append('')
                recent_post = recent_post[
                    :min(len(recent_post), post_limit)
                ]
                merged_topic = [0.0] * 20
                if not recent_post_topic:
                    merged_topic = [0.05] * 20
                else:
                    recent_post_topic = recent_post_topic[
                        :min(len(recent_post_topic), post_limit)
                    ]
                    for v in recent_post_topic:
                        merged_topic = self.plus_topic_vector(
                            merged_topic, 
                            v
                        )
                merged_topic = ' '.join(
                    [str(val / max(1, num)) for val in merged_topic]
                )
                recent_post = ' '.join(recent_post)
                cand_post = thread2initial_post[thread_id]
                for c in ['\n', '\r', '\t']:
                    recent_post = recent_post.replace(c, ' ')
                    cand_post = cand_post.replace(c, ' ')
                if not cand_post:
                    print(thread_id, reply_user_name, time, label)
                    continue
                recent_post = \
                    recent_post[
                        :min(len(recent_post), self.excel_size_limit)
                    ]
                cand_post = \
                    cand_post[
                        :min(len(cand_post), self.excel_size_limit)
                    ]
                dataset.append(
                    [
                        reply_user_name, 
                        thread_id, 
                        recent_post, 
                        cand_post, 
                        merged_topic, 
                        cand_topic, 
                        time, 
                        train_or_test,
                        label,
                        time_dif + 1
                    ]
                )
                if train_or_test == self.train_name:
                    if label == 1:
                        train_pos += 1
                    else:
                        train_neg += 1
                else:
                    if label == 1:
                        test_pos += 1
                    else:
                        test_neg += 1
                cnt+=1
                if cnt%10000==0:
                    print(cnt)
        print('generate done')
        # dataset.sort(key=lambda v : v[-4])
        print('Successfully generate dataset!')
        print('Positive sample number in training data: {}\n'.format(train_pos)
        + 'Negative sample number in training data: {}\n'.format(train_neg)
        + 'Positive ratio in training data: {}\n'.format(train_pos / train_neg)
        + 'Positive sample number in testing data: {}\n'.format(test_pos)
        + 'Negative sample number in testing data: {}\n'.format(test_neg)
        + 'Positive ratio in testing data: {}\n'.format(test_pos / test_neg))
        return dataset

    def gen_dataset_based_on_time(
            self,
            id2topic,
            train_samples,
            test_samples,
            user2all_thread,
            thread2initial_post,
            post_limit,
            experiment_data_path,
            data_name

            ):
        train_pos = test_pos = train_neg = test_neg = 0
        #generate dataset
        cnt=0
        with open(
                experiment_data_path + '{}.csv'.format(data_name), 
                mode='a'
        ) as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(self.attributes)
        for samples, train_or_test in [
            [train_samples, self.train_name],
            [test_samples, self.test_name]
        ]:
            for thread_id, reply_user_name, time, label in samples:
                int_thread_id = int(thread_id)
                assert(int_thread_id in id2topic)
                cand_topic = ' '.join(
                    [str(val) for val in id2topic[int_thread_id]]
                )
                idx = bisect.bisect_left(
                    user2all_thread[reply_user_name], 
                    (time, '')
                )
                recent_post = []
                recent_post_topic = []
                num = 0
                time_dif = int(1e9)
                for i in range(idx - 1, -1, -1):
                    if user2all_thread[reply_user_name][i][1] == thread_id:
                        continue
                    recent_thread_time, recent_thread_id = \
                        user2all_thread[reply_user_name][i]
                    time_dif = min(
                        time_dif, 
                        int(
                            (
                                time - recent_thread_time
                            ).total_seconds() / 60
                        )
                    )
                    post_text = thread2initial_post[recent_thread_id]
                    if len(recent_post) + len(post_text) \
                        > self.excel_size_limit:
                        continue
                    int_recent_thread_id = int(recent_thread_id)
                    recent_post.append(post_text)
                    assert(int_recent_thread_id in id2topic)
                    recent_post_topic.append(id2topic[int_recent_thread_id])
                    num += 1
                
                while len(recent_post) < post_limit:
                    recent_post.append('')
                recent_post = recent_post[
                    :min(len(recent_post), post_limit)
                ]
                merged_topic = [0.0] * 20
                if not recent_post_topic:
                    merged_topic = [0.05] * 20
                else:
                    recent_post_topic = recent_post_topic[
                        :min(len(recent_post_topic), post_limit)
                    ]
                    for v in recent_post_topic:
                        merged_topic = self.plus_topic_vector(
                            merged_topic, 
                            v
                        )
                merged_topic = ' '.join(
                    [str(val / max(1, num)) for val in merged_topic]
                )
                recent_post = ' '.join(recent_post)
                cand_post = thread2initial_post[thread_id]
                for c in ['\n', '\r', '\t']:
                    recent_post = recent_post.replace(c, ' ')
                    cand_post = cand_post.replace(c, ' ')
                if not cand_post:
                    print(thread_id, reply_user_name, time, label)
                    continue
                recent_post = \
                    recent_post[
                        :min(len(recent_post), self.excel_size_limit)
                    ]
                cand_post = \
                    cand_post[
                        :min(len(cand_post), self.excel_size_limit)
                    ]
                with open(experiment_data_path + '{}.csv'.format(data_name), mode='a') as csv_file:
                    tuple_writer = csv.writer(
                        csv_file, 
                        delimiter=',', 
                        quotechar='"', 
                        quoting=csv.QUOTE_MINIMAL
                    )
                    tuple_writer.writerow(
                        [
                        reply_user_name, 
                        thread_id, 
                        recent_post, 
                        cand_post, 
                        merged_topic, 
                        cand_topic, 
                        time, 
                        train_or_test,
                        label,
                        time_dif + 1
                    ])
                # dataset.append(
                #     [
                #         reply_user_name, 
                #         thread_id, 
                #         recent_post, 
                #         cand_post, 
                #         merged_topic, 
                #         cand_topic, 
                #         time, 
                #         train_or_test,
                #         label,
                #         time_dif + 1
                #     ]
                # )
                if train_or_test == self.train_name:
                    if label == 1:
                        train_pos += 1
                    else:
                        train_neg += 1
                else:
                    if label == 1:
                        test_pos += 1
                    else:
                        test_neg += 1
                cnt+=1
                if cnt%10000==0:
                    print(cnt)
        print('generate done')
        # dataset.sort(key=lambda v : v[-4])
        print('Successfully generate dataset!')
        print('Positive sample number in training data: {}\n'.format(train_pos)
        + 'Negative sample number in training data: {}\n'.format(train_neg)
        + 'Positive ratio in training data: {}\n'.format(train_pos / train_neg)
        + 'Positive sample number in testing data: {}\n'.format(test_pos)
        + 'Negative sample number in testing data: {}\n'.format(test_neg)
        + 'Positive ratio in testing data: {}\n'.format(test_pos / test_neg))
        # return dataset

    
    
    def save_csv(self, data, data_path, attributes):
        with open(
                data_path, 
                mode='w'
        ) as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(attributes)
            for i in range(len(data)):
                tuple_writer.writerow(data[i])
        print('Successfully saved {}'.format(data_path))
    
    
    def get_dataset_for_CLIR(
            self, 
            data_path, 
            decision_making_thread_id_list, 
            experiment_data_path, 
            id_topic_vector_path, 
            train_ratio,
            train_sample_num,
            test_sample_num,
            cold_start_user,
            cold_start_thread,
            data_name,
            post_limit,
            user_limit,
            train_thread_limit,
            test_thread_limit,
            random_sample_threads,
            random_sample_users,
            use_ratio,
            time_oblivious
    ):
        #load data
        data = self.load_data(data_path)
        decision_making_threads = set(
            self.get_decision_making_thread_id_list(
                decision_making_thread_id_list
            )
        )
        id2topic = self.load_id_to_topic(id_topic_vector_path)
        print('Successfully load the data!')
        
        #collect data
        thread_id_time, \
        user2reply_thread, \
        user2all_thread, \
        thread2initial_post, \
        thread2initial_author, \
        user_adj = self.collect_data(data, decision_making_threads)
        
        #generate positive/negative samples
        thread_num = len(thread_id_time)
        if use_ratio:
            last_idx = int(train_ratio * thread_num) - 1
        else:
            last_idx = train_thread_limit - 1
        time_th = thread_id_time[last_idx][1]
        print('Time threshold = {}'.format(time_th))
        
        #filter out users with at least 5 replies
        all_users = self.gen_all_users(user2reply_thread)
        
        #split old/new threads
        old_thread, new_thread = self.gen_old_new_threads(
            thread_id_time, 
            last_idx,
            train_thread_limit,
            test_thread_limit,
            random_sample_threads,
            time_oblivious
        )
        
        #generate train/test samples
        train_samples, test_samples = self.gen_train_test_sample(
            all_users,
            old_thread,
            new_thread,
            user2reply_thread,
            user2all_thread,
            thread2initial_author,
            train_sample_num,
            test_sample_num,
            time_th,
            cold_start_user,
            cold_start_thread,
            user_limit,
            random_sample_users
        )
        
        #generate dataset
        dataset = self.gen_dataset(
            id2topic,
            train_samples,
            test_samples,
            user2all_thread,
            thread2initial_post,
            post_limit
        )
        
        #save dataset
        self.save_csv(
            dataset, 
            experiment_data_path + '{}.csv'.format(data_name), 
            self.attributes
        )
        
        #save user mapping to their replied threads
        self.save_csv(
            [
                [user_name, '|'.join([str(v[1]) for v in st])]
                for user_name, st in user2reply_thread.items()
                    if st
            ],
            experiment_data_path + 'user_to_reply_thread.csv',
            ['user_name', 'threads']
        )
        
        #save thread id mapping to initial post
        self.save_csv(
            [
                [thread_id, initial_post]
                for thread_id, initial_post in thread2initial_post.items()
            ],
            experiment_data_path + 'thread_to_initial_post.csv',
            ['thread_id', 'initial_post']
        )
        
        #save user adjacency graph
        self.save_csv(
            [
                [user_name, '|'.join(list(st))]
                for user_name, st in user_adj.items()
            ],
            experiment_data_path + 'user_adj.csv',
            ['user_name', 'neighbors']
        )
        
        #save user thread id mapping to replied users
        self.save_thread2_user_list(
            experiment_data_path + 'user_to_reply_thread.csv',
            experiment_data_path + 'thread_to_replied_user.csv', 
        )
        
        
    def get_user_thread_pairs(self, data_path):
        data = pd.read_csv(data_path)
        user2thread2train_label = {}
        for i in range(len(data)):
            user_id = data['user_id'][i]
            thread_id = data['thread_id'][i]
            train_or_test = data['train_or_test'][i]
            label = data['label'][i]
            if user_id not in user2thread2train_label:
                user2thread2train_label[user_id] = {}
            user2thread2train_label[user_id][thread_id] \
              = [train_or_test, label]
        return user2thread2train_label
    
    
    def gen_dataset_with_time(
            self,
            user2all_thread,
            user2reply_thread,
            thread_id_time,
            user2thread2train_label,
            id2topic,
            thread2initial_post,
            post_limit
    ):
        #generate training data
        samples = []
        thread_num = len(thread_id_time)
        for user_id in user2all_thread:
            if user_id not in user2thread2train_label:
                continue
            for i in range(thread_num):
                thread_id, time = thread_id_time[i]
                thread_id = int(thread_id)
                if thread_id not in user2thread2train_label[user_id]:
                    continue
                samples.append(
                    (
                        thread_id, 
                        user_id, 
                        time, 
                        user2thread2train_label[user_id][thread_id][0],
                        user2thread2train_label[user_id][thread_id][1]
                    )
                )  
        print('Train and test data generated successfully!')
        dataset = []
        train_pos = test_pos = train_neg = test_neg = 0
        for thread_id, reply_user_name, time, train_or_test, label in samples: 
            #collect reply users' five most recent 
            #posts (reply or initial) close to reply time
            assert(thread_id in id2topic)
            cand_topic = ' '.join(
                [str(val) for val in id2topic[thread_id]]
            )
            idx = bisect.bisect_left(
                user2all_thread[reply_user_name], (time, '')
            )
            recent_post = []
            recent_post_topic = []
            time_diff = []
            for i in range(idx - 1, -1, -1):
                if user2all_thread[reply_user_name][i][1] == thread_id:
                    continue
                recent_thread_id = user2all_thread[reply_user_name][i][1]
                post_text = thread2initial_post[recent_thread_id]
                if len(recent_post) + len(post_text) > self.excel_size_limit:
                    continue
                int_recent_thread_id = int(recent_thread_id)
                recent_post.append(post_text)
                time_diff.append(
                    str(
                        int(
                            (
                                time - user2all_thread[reply_user_name][i][0]
                            ).total_seconds() / 60
                        )
                    )
                )
                assert(int_recent_thread_id in id2topic)
                recent_post_topic.append(
                    ' '.join(
                        [str(val) for val in id2topic[int_recent_thread_id]]
                    )
                )
            
            while len(recent_post) < post_limit:
                recent_post.append('')
                recent_post_topic.append(
                    ' '.join(['0.05' for _ in range(20)])
                )
                time_diff.append(str(10**9))
            recent_post = recent_post[
                :min(len(recent_post), post_limit)
            ]
            recent_post_topic = recent_post_topic[
                :min(len(recent_post_topic), post_limit)
            ]
            time_diff = time_diff[:min(len(time_diff), post_limit)]
            time_dif_val = min([int(s) for s in time_diff]) + 1
            if len(recent_post) != post_limit \
                or len(recent_post_topic) != post_limit:
                continue
            #reverse to make the most recent at the end
            recent_post.reverse()
            recent_post_topic.reverse()
            time_diff.reverse()
            cand_post = thread2initial_post[str(thread_id)]
            for c in ['\n', '\r', '\t']:
                for i in range(len(recent_post)):
                    recent_post[i] = recent_post[i].replace(c, ' ')
                cand_post = cand_post.replace(c, ' ')
            if not cand_post:
                print(thread_id, reply_user_name, time, train_or_test, label)
                continue
            recent_post = self.encode(recent_post)
            recent_post_topic = '|'.join(recent_post_topic)
            time_diff = '|'.join(time_diff)
            recent_post = recent_post[
                :min(len(recent_post), self.excel_size_limit)
            ]
            cand_post = cand_post[
                :min(len(cand_post), self.excel_size_limit)
            ]
            dataset.append(
                    [
                        reply_user_name, 
                        thread_id, 
                        recent_post, 
                        cand_post, 
                        recent_post_topic, 
                        cand_topic, 
                        time_diff, 
                        train_or_test,
                        label,
                        time_dif_val
                    ]
                )
            if train_or_test == self.train_name:
                if label == 1:
                    train_pos += 1
                else:
                    train_neg += 1
            else:
                if label == 1:
                    test_pos += 1
                else:
                    test_neg += 1
        print('Successfully generate dataset!')
        print('Positive sample number in training data: {}\n'.format(train_pos)
        + 'Negative sample number in training data: {}\n'.format(train_neg)
        + 'Positive ratio in training data: {}\n'.format(train_pos / train_neg)
        + 'Positive sample number in testing data: {}\n'.format(test_pos)
        + 'Negative sample number in testing data: {}\n'.format(test_neg)
        + 'Positive ratio in testing data: {}\n'.format(test_pos / test_neg))
        return dataset
    
    
    def get_dataset_for_CLLIR(
            self, 
            data_path, 
            decision_making_thread_id_list, 
            experiment_data_path, 
            id_topic_vector_path, 
            train_ratio, 
            post_limit,
            data_name
    ):
        #load data
        data = self.load_data(data_path)
        decision_making_threads = set(
            self.get_decision_making_thread_id_list(
                decision_making_thread_id_list
            )
        )
        id2topic = self.load_id_to_topic(id_topic_vector_path)
        user2thread2train_label = self.get_user_thread_pairs(
            experiment_data_path + data_name + '.csv'
        )
        print('Successfully load the data!')
        
        #collect data
        thread_id_time, \
        user2reply_thread, \
        user2all_thread, \
        thread2initial_post, \
        thread2initial_author, \
        user_adj = self.collect_data(data, decision_making_threads)
        
        # #generate positive/negative samples
        # thread_num = len(thread_id_time)
        # last_idx = int(train_ratio * thread_num) - 1
        # time_th = thread_id_time[last_idx][1]
        # print('Time threshold = {}'.format(time_th))
        
        #generate dataset
        dataset = self.gen_dataset_with_time(
            user2all_thread,
            user2reply_thread,
            thread_id_time,
            user2thread2train_label,
            id2topic,
            thread2initial_post,
            post_limit
        )
        
        #save dataset
        self.save_csv(
            dataset, 
            experiment_data_path + data_name + '_with_time.csv', 
            self.attributes
        )
        
        #save user mapping to their replied threads
        self.save_csv(
            [
                [user_name, '|'.join([str(v[1]) for v in st])]
                for user_name, st in user2reply_thread.items()
                    if st
            ],
            experiment_data_path + 'user_to_reply_thread.csv',
            ['user_name', 'threads']
        )
        
        #save thread id mapping to initial post
        self.save_csv(
            [
                [thread_id, initial_post]
                for thread_id, initial_post in thread2initial_post.items()
            ],
            experiment_data_path + 'thread_to_initial_post.csv',
            ['thread_id', 'initial_post']
        )
        
        #save user adjacency graph
        self.save_csv(
            [
                [user_name, '|'.join(list(st))]
                for user_name, st in user_adj.items()
            ],
            experiment_data_path + 'user_adj.csv',
            ['user_name', 'neighbors']
        )
        
        
    def gen_user_cand_threads(
            self, 
            whole_data_path, 
            sample_data_path, 
            user_num
    ):
        data = pd.read_csv(whole_data_path, encoding = "ISO-8859-1")
        print('Successfully load the data!')
        pos_num = sum(data['label'].values)
        user_set = set()
        while len(user_set) < user_num:
            idx = random.randrange(pos_num, len(data))
            user_set.add(data['user_id'][idx])
        print('Successfully generate the user set!')
        sample_data = []
        for idx in range(pos_num * 2):
            if data['user_id'][idx] in user_set:
                sample_data.append(
                    [
                        data[attribute][idx] 
                        for attribute in self.attributes
                    ]
                )
        with open(sample_data_path, mode='w') as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(self.attributes)
            for v in sample_data:
                tuple_writer.writerow(v)
        print('Successfully saved the user and candidate thread dataset')
           
        
    def get_topic_vector(self, topic_vector_path):
        fp = open(topic_vector_path, "r")
        lines = fp.readlines()
        v = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            idx += 1
            while idx < len(lines) and lines[idx][0] != 'D':
                line += lines[idx]
                idx += 1
            start = line.find('[') + 1
            end = line.find(']')
            v.append(line[start:end].split())
        return v
    
    
    def gen_id_to_topic(
            self, 
            decision_making_thread_id_list, 
            topic_vector_path, 
            id_topic_vector_path
    ):
        decision_making_threads = self.get_decision_making_thread_id_list(
            decision_making_thread_id_list
        )
        topic2vector = self.get_topic_vector(topic_vector_path)
        print(len(decision_making_threads), len(topic2vector))
        assert(len(decision_making_threads) == len(topic2vector))
        with open(id_topic_vector_path, mode='w') as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(['thread_id', 'topic_vector'])
            for i in range(len(decision_making_threads)):
                tuple_writer.writerow(
                    [decision_making_threads[i], ' '.join(topic2vector[i])]
                )
        print('Successfully generate id to topic mapping')
        
        
    def load_id_to_topic(self, id_topic_vector_path):
        mp = {}
        data = pd.read_csv(id_topic_vector_path, encoding = "ISO-8859-1")
        for i in range(len(data)):
            v = data['topic_vector'][i].split()
            assert(len(v) == 20)
            mp[data['thread_id'][i]] = [float(s) for s in v]
        return mp
        
    
    def gen_random_sample(
            self, 
            sample_data_path, 
            experiment_data_path, 
            random_sample_data_path
    ):
        user_to_reply_thread_data = pd.read_csv(
            experiment_data_path + 'user_to_reply_thread.csv', 
            encoding = "ISO-8859-1"
        )
        sample_data = pd.read_csv(sample_data_path, encoding = "ISO-8859-1")
        thread_data = pd.read_csv(
            experiment_data_path + 'thread_to_initial_post.csv', 
            encoding = "ISO-8859-1"
        )
        user_to_reply_thread = {}
        for i in range(len(user_to_reply_thread_data)):
            user_to_reply_thread[user_to_reply_thread_data['user_name'][i]] \
                = set(user_to_reply_thread_data['threads'][i].split('|'))
        for i in range(4000, 4000 + 3694):
            if sample_data['label'][i] != 0:
                raise 'Error {}'.format(str(i))
            user_id = sample_data['user_id'][i]
            idx = random.randrange(len(thread_data))
            thread_id = thread_data['thread_id'][idx]
            while thread_id in user_to_reply_thread[user_id]:
                idx = random.randrange(len(thread_data))
                thread_id = thread_data['thread_id'][idx]
            sample_data['thread_id'][i] = thread_id
            sample_data['initial_post_to_recommend'][i] = \
                thread_data['initial_post'][idx]
        with open(random_sample_data_path, mode='w') as csv_file:
            tuple_writer = csv.writer(
                csv_file, 
                delimiter=',', 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )
            tuple_writer.writerow(self.attributes)
            for i in range(len(sample_data)):
                tuple_writer.writerow(
                    [
                        sample_data[att][i] 
                        for att in self.attributes
                    ]
                )
        print('Successfully saved the random dataset')
        
        
    def gen_initial_posts(self, data_path, data_folder_path, user_all_threads):
        data = self.load_data(data_path)
        print('Successfully load the data!')
        posts = []
        ids = []
        for i in range(len(data)):
            if len(data[i]['posts']) == 0:
                print(data[i]['id'])
                continue
            posts.append(data[i]['posts'][0]['post'])
            ids.append(data[i]['id'])
        with open(data_folder_path + 'all_posts_intial.txt', 'w') as f:
            for item in posts:
                f.write("%s\n" % item)
        print('Successfully saved inital post text!')
        #consider all threads
        if user_all_threads:
            with open(data_folder_path + 'idresult.txt', 'w') as f:
                for item in ids:
                    f.write("%s\n" % item)
        print('Successfully saved id results!')
        
        
    def gen_csv_from_json(self, json_data_path, csv_data_path):
        data = self.load_data(json_data_path)
        with open(csv_data_path, mode='w') as csv_file:
            fieldnames = ['thread_id', 'init_post', 'replies']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(data)):
                if not data[i]['posts']:
                    continue
                thread_id = data[i]["id"]
                init_post = data[i]['posts'][0]['post']
                replies = ' '.join(
                    [
                        data[i]['posts'][j]['post'] 
                        for j in range(1, len(data[i]['posts']))
                    ]
                )
                writer.writerow(
                    {
                        'thread_id': thread_id, 
                        'init_post': init_post, 
                        'replies': replies
                    }
                )
        
        
    def get_statistic_of_thread_recommendation(self, sample_data_path):
        data = pd.read_csv(sample_data_path, encoding = "ISO-8859-1")
        user_st = set([e for e in data['user_id'].values])
        thread_st = set([e for e in data['thread_id'].values])
        tot_user = len(user_st)
        tot_thread = len(thread_st)
        print('User number: {}'.format(len(user_st)))
        print('Thread number: {}'.format(len(thread_st)))
        train_word_len = 0
        test_word_len = 0
        train_num = 0
        test_num = 0
        positive_num = 0
        train_users = set()
        train_threads = set()
        test_users = set()
        test_threads = set()
        for i in range(len(data)):
            user_id = data['user_id'][i]
            thread_id = data['thread_id'][i]
            positive_num += data['label'][i] == 1
            if data['train_or_test'][i] == self.train_name:
                if user_id in user_st:
                    user_st.remove(user_id)
                if thread_id in thread_st:
                    thread_st.remove(thread_id)
                train_word_len += \
                    len(data['initial_post_to_recommend'][i].split())
                train_num += 1
                train_users.add(user_id)
                train_threads.add(thread_id)
            else:
                test_word_len += \
                    len(data['initial_post_to_recommend'][i].split())
                test_num += 1
                test_users.add(user_id)
                test_threads.add(thread_id)
        print('New user number: {}\n'.format(1.0 * len(user_st) / tot_user)
        + 'New thread number: {}\n'.format(1.0 * len(thread_st) / tot_thread)
        + 'Train number: {}\n'.format(train_num)
        + 'Test number: {}\n'.format(test_num)
        + 'Average train word number: {}\n'.format(
                1.0 * train_word_len / train_num
            )
        + 'Average test word number: {}\n'.format(
                1.0 * test_word_len / test_num
            )
        + 'Sparsity is {}\n'.format(
                1 - positive_num / tot_user / tot_thread
            ))
        print('Train user number: {}'.format(len(train_users)))
        print('Test user number: {}'.format(len(test_users)))
        print('Train thread number: {}'.format(len(train_threads)))
        print('Test thread number: {}'.format(len(test_threads)))
        
        
    def prepare_data(self, data, columns):
        for name in columns:
            data[name] = data[name].apply(
                lambda x: x.lower() if type(x) == type('') else str(x)
            )
            data[name] = data[name].apply(
                lambda x: re.sub('[^a-zA-z0-9\s]', '', x)
            )
        print('Finish preprocessing the text!')
        return data
        
    
    def get_token(
            self,
            token_source_path, 
            token_file_path, 
            token_source_columns,
            max_features
    ):
        if path.exists(token_file_path):
            with open(token_file_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            print('Successfuly load tokenizer!')
            return tokenizer
        token_source_data  = pd.read_csv(
            token_source_path, 
            encoding = "ISO-8859-1"
        )
        token_source_data = self.prepare_data(
            token_source_data, 
            token_source_columns
        )
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        t = []
        for column in token_source_columns:
            t += [s for s in token_source_data[column].values]
        tokenizer.fit_on_texts(t)
        with open(token_file_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
        print('Successfully generate tokenizer!')
        return tokenizer
    
    
    def get_question_probability(self, sentence):
        q = 0
        q += 0.7 if '?' in sentence else 0
        valid = False
        for word in self.question_word_set:
            if word in sentence:
                valid = True
                break
        if valid:
            q += 0.2
        valid = False
        for word in self.question_word_set:
            if sentence.startswith(word):
                valid = True
                break
        if valid:
            q += 0.1
        return q
    
    
    def get_action_probability(self, sentence):
        text = word_tokenize(sentence)
        tagged = pos_tag(text)
        if len(
                [
                    word for word in tagged 
                    if word[1] in ["VBC", "VBF", "MD"]
                ]
            ) > 0:
            return 1.0
        return 0
    
    
    def get_question_action(self, data):
        question = []
        action = []
        for i in range(len(data)):
            if not data['initial_author_reply'][i] \
                or str(data['initial_author_reply'][i]) == 'nan':
                question.append(0.0)
                action.append(0.0) 
                continue
            list_sentence_test = nltk.sent_tokenize(
                data['initial_author_reply'][i]
            )
            question_p = 0.5
            action_p = 0.5
            for sentence in list_sentence_test:
                question_p = max(
                    question_p,
                    self.get_question_probability(sentence)
                )
                action_p = max(
                    action_p,
                    self.get_action_probability(sentence)
                )
            question.append(question_p)
            action.append(action_p) 
        print('Question and action probability is calculated!')
        return question, action
    
    
    def get_question_action_max(self, question, action):
        question_action_max = []
        for i in range(len(question)):
            question_action_max.append(max(question[i], action[i]))
        print("Find max question and action probability!")
        return question_action_max
        
    
    def get_seq_input(self,data, tokenizer, influence_columns, maxlen):
        posts = [
            pad_sequences(
                tokenizer.texts_to_sequences(data[column].values), 
                maxlen
            )
            for column in influence_columns
        ]
        print('Sequence input generated successfully!')
        return posts
    
    
    def gen_input_to_model(self, train_data, test_data, train_Y):
        train_data = [
            np.asarray(
                [
                    train_data[i][j]
                    for i in range(len(train_data))
                ]
            )
            for j in range(len(train_data[0]))
        ]
        test_data = [
            np.asarray(
                [
                    test_data[i][j]
                    for i in range(len(test_data))
                ]
            )
            for j in range(len(test_data[0]))
        ]
        return train_data, test_data, \
                np.asarray([[val, 1 - val] for val in train_Y])
        
        
    def load_model(self, model_path):
        json_file  = open(model_path+'model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_path+'model.h5')
        loaded_model.compile(
            loss = 'binary_crossentropy', 
            optimizer = 'adam', 
            metrics = ['accuracy']
        )
        print('Model is loaded successfully!')
        return loaded_model 
        
        
    def get_influence_input(
        self,
        relevance_model, 
        posts, 
        Y, 
        question, 
        action, 
        train_idxs,
        batch_size,
        output_dim
    ):
        posts_train = [
            np.asarray(
                [
                    posts[i][j][:]
                    for j in range(len(posts[i]))
                        if j in train_idxs
                ]
            )
            for i in range(len(posts))
        ]
        posts_test = [
            np.asarray(
                [
                    posts[i][j][:]
                    for j in range(len(posts[i]))
                        if j not in train_idxs
                ]
            )
            for i in range(len(posts))
        ]
        train12 = relevance_model.predict(
            [posts_train[0], posts_train[1]], 
            batch_size=batch_size, 
            verbose=output_dim
        )
        train23 = relevance_model.predict(
            [posts_train[1], posts_train[2]], 
            batch_size=batch_size, 
            verbose=output_dim
        ) 
        test12 = relevance_model.predict(
            [posts_test[0], posts_test[1]], 
            batch_size=batch_size, 
            verbose=output_dim
        )
        test23 = relevance_model.predict(
            [posts_test[1], posts_test[2]], 
            batch_size=batch_size, 
            verbose=output_dim
        ) 
        train_Y = np.asarray([
            [Y[i], 1 - Y[i]] 
            for i in range(len(Y)) 
                if i in train_idxs
        ])
        test_Y = [
            Y[i] 
            for i in range(len(Y)) 
            if i not in train_idxs
        ]
        train_question = np.asarray([
            question[i]
            for i in range(len(question))
                if i in train_idxs
        ])
        test_question = np.asarray([
            question[i]
            for i in range(len(question))
                if i not in train_idxs
        ])
        train_action = np.asarray([
            action[i]
            for i in range(len(action))
                if i in train_idxs
        ])
        test_action = np.asarray([
            action[i]
            for i in range(len(action))
                if i not in train_idxs
        ])
        print('Influence input is calcualted')
        return train12, train23, \
                test12, test23, \
                train_question, test_question, \
                train_action, test_action, \
                train_Y, test_Y
        
        
    def save_data(self, data, data_path):
        with open(data_path, 'w') as f:
            for item in data:
                f.write("%s\n" % item)
        print('Data saved successfully!')
        
        
    def create_dirs(self, dir_list):
        for model_dir_name in dir_list:
            try:
                mkdir(model_dir_name)
                print("Directory " , model_dir_name ,  " Created ") 
            except FileExistsError:
                print("Directory " , model_dir_name ,  " already exists")
        
        
    def load_idx(self, idxs_path):
        idxs = np.loadtxt(idxs_path, delimiter='\n', dtype=int)
        print('Load index successfully!')
        return idxs
    
    
    def str2arr(self, s, maxlen):
        s = s[1:-1]
        s = s.split()
        assert(len(s) == maxlen)
        return [int(val) for val in s]
    
    
    def load_relevant_pairs(self, relevant_pair_path, maxlen):
        relevance_data = pd.read_csv(relevant_pair_path)
        num = len(relevance_data)
        relevance_train_data = [
            [
                self.str2arr(relevance_data['post1'][i], maxlen),
                self.str2arr(relevance_data['post2'][i], maxlen)
            ]
            for i in range(num)
                if relevance_data['train_or_test'][i] == self.train_name
        ]
        relevance_test_data = [
            [
                self.str2arr(relevance_data['post1'][i], maxlen),
                self.str2arr(relevance_data['post2'][i], maxlen)
            ]
            for i in range(num)
                if relevance_data['train_or_test'][i] == self.test_name
        ]
        relevance_train_Y = [
            relevance_data['label'][i]
            for i in range(num)
                if relevance_data['train_or_test'][i] == self.train_name
        ]
        relevance_test_Y = [
            relevance_data['label'][i]
            for i in range(num)
                if relevance_data['train_or_test'][i] == self.test_name
        ]
        print('Load relevant pairs successfully!')
        return relevance_train_data, relevance_test_data, \
                relevance_train_Y, relevance_test_Y
   
                
    def load_thread2initial_author(self, thread2intial_author_path):
        thread2initial_author_data = pd.read_csv(
            thread2intial_author_path, 
            encoding = "ISO-8859-1"
        )
        thread2initial_author = {}
        for i in range(len(thread2initial_author_data)):
            thread2initial_author[
                int(thread2initial_author_data['thread_id'][i])
            ] = thread2initial_author_data['user_name'][i]
        return thread2initial_author
    
                
    def load_user2reply_thread(self, user2reply_thread_path):
        user2reply_thread_data = pd.read_csv(
            user2reply_thread_path, 
            encoding = "ISO-8859-1"
        )
        user2reply_thread = {}
        for i in range(len(user2reply_thread_data)):
            user2reply_thread[user2reply_thread_data['user_name'][i]] = \
                set(user2reply_thread_data['threads'][i].split('|'))
        return user2reply_thread
    
    
    def get_recommendation_input(
            self, 
            data, 
            tokenizer, 
            thread_recommendation_content_columns,
            thread_recommendation_topic_columns,
            maxlen
    ):
        data = self.prepare_data(
            data, 
            thread_recommendation_content_columns
        )    
        target_users_recent_posts, \
        initial_post_to_recommend = \
        self.get_seq_input(
            data, 
            tokenizer, 
            thread_recommendation_content_columns, 
            maxlen
        )
        target_users_topic_vector, \
        candidate_thread_topic_vector = [
            [
                [float(val) for val in s.split()] 
                for s in data[column_name].values
            ]
            for column_name in thread_recommendation_topic_columns
        ]
        Y = data['label'].values
        time = data['time'].values
        train_or_test = data['train_or_test'].values
        
        print('Recommendation input generated successfully!')
        return target_users_recent_posts, \
                initial_post_to_recommend, \
                target_users_topic_vector, \
                candidate_thread_topic_vector, \
                Y, time, \
                train_or_test
   
    
    def split_recommendation_data(
            self,
            data,
            post1_text, post2_text, 
            post1_topic, post2_topic, 
            Y, time, train_or_test
    ):
        #balance dataset
        post1_text_train = np.asarray([
            post1_text[i]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name
        ])
        post1_text_test = np.asarray([
            post1_text[i]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        post2_text_train = np.asarray([
            post2_text[i]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name
        ])
        post2_text_test = np.asarray([
            post2_text[i]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        post1_topic_train = np.asarray([
            post1_topic[i]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name
        ])
        post1_topic_test = np.asarray([
            post1_topic[i]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        post2_topic_train = np.asarray([
            post2_topic[i]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name
        ])
        post2_topic_test = np.asarray([
            post2_topic[i]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        Y_test = np.asarray([
            Y[i]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        Y_train_input = np.asarray([
            [Y[i], 1 - Y[i]]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name
        ])
        
        user_id2idx = {}
        for i in range(len(Y)):
            if data['user_id'][i] not in user_id2idx:
                user_id2idx[data['user_id'][i]] = len(user_id2idx) + 1
                
        train_user = set(
            [
                data['user_id'][i]
                for i in range(len(Y))
                    if train_or_test[i] == self.train_name
            ]
        )
        test_user = set(
            [
                data['user_id'][i]
                for i in range(len(Y))
                    if train_or_test[i] != self.train_name
            ]
        )
        train_thread = set(
            [
                data['thread_id'][i]
                for i in range(len(Y))
                    if train_or_test[i] == self.train_name 
            ]
        )
        test_thread = set(
            [
                data['thread_id'][i]
                for i in range(len(Y))
                    if train_or_test[i] != self.train_name
            ]
        )
        user_thread_train = [
            [data['user_id'][i], data['thread_id'][i]]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name
        ]
        user_thread_test = [
            [data['user_id'][i], data['thread_id'][i]]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ]
        # user_ids_train = np.asarray([
        #     zlib.crc32(data['user_id'][i].encode('utf-8'))
        #     for i in range(len(Y))
        #         if train_or_test[i] == self.train_name
        # ])
        # user_ids_test = np.asarray([
        #     zlib.crc32(data['user_id'][i].encode('utf-8'))
        #     for i in range(len(Y))
        #         if train_or_test[i] != self.train_name
        # ])
        user_ids_train = np.asarray([
            user_id2idx[data['user_id'][i]]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name 
        ])
        user_ids_test = np.asarray([
            user_id2idx[data['user_id'][i]]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        time_dif_train = np.asarray([
            int(data['time_dif'][i])
            for i in range(len(Y))
                if train_or_test[i] == self.train_name 
        ])
        time_dif_test = np.asarray([
            int(data['time_dif'][i])
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        print('Train user number: {}'.format(len(train_user)))
        print('Test user number: {}'.format(len(test_user)))
        print('Overlap user number: {}'.format(len(train_user & test_user)))
        print('Train thread number: {}'.format(len(train_thread)))
        print('Test thread number: {}'.format(len(test_thread)))
        print('Overlap thread number: {}'.format(
            len(train_thread & test_thread))
        )
        pos_train = sum([v[0] for v in Y_train_input])
        neg_train = len(Y_train_input) - pos_train
        pos_test = sum(Y_test)
        neg_test = len(Y_test) - pos_test
        print(
            'Positive sample number in training set: {}\n'.format(pos_train)
            +'Negative sample number in training set: {}\n'.format(neg_train) 
            +'Positive sample number in testing set: {}\n'.format(pos_test)
            + 'Negative sample number in testing set: {}\n'.format(neg_test))
        print('Data split successfully!')
        return post1_text_train, post1_text_test, \
                post2_text_train, post2_text_test, \
                post1_topic_train, post1_topic_test, \
                post2_topic_train, post2_topic_test, \
                Y_train_input, Y_test, \
                user_ids_train, user_ids_test, \
                user_thread_train, user_thread_test, \
                time_dif_train, time_dif_test
                
    def get_recommendation_time_input(
            self,
            data, 
            tokenizer,
            maxlen,
            post_limit
    ):
        #decode
        target_users_recent_posts = [
            self.decode(s) 
            for s in data['target_users_recent_posts'].values
        ]
        initial_post_to_recommend = [
            s for s in data['initial_post_to_recommend'].values
        ]
        #preprocessing
        target_users_recent_posts = [
            [
                re.sub('[^a-zA-z0-9\s]', '', s.lower()) 
                for s in v
            ]
            for v in target_users_recent_posts
        ]
        initial_post_to_recommend = [
            re.sub('[^a-zA-z0-9\s]', '', s.lower()) 
            for s in initial_post_to_recommend
        ]
        target_users_recent_posts = [
            pad_sequences(tokenizer.texts_to_sequences(v), maxlen) 
            for v in target_users_recent_posts
        ]
        initial_post_to_recommend = \
            pad_sequences(
                tokenizer.texts_to_sequences(initial_post_to_recommend), 
                maxlen
            ) 
        print('Successfully generate text sequence')
        target_users_topic_vector = [
            [
                [
                    float(val) for val in vals.split()
                ] 
                for vals in s.split('|')
            ] 
            for s in data['target_users_topic_vector'].values
        ]
        candidate_thread_topic_vector = [
            [
                float(val) for val in s.split()
            ] 
            for s in data['candidate_thread_topic_vector'].values
        ]
        print('Successfully generate topic vector')
        Y = data['label'].values
        time = [[float(val) for val in s.split('|')] for s in data['time']]
        # #normalization
        # for i in range(len(time)):
        #     max_val = max(time[i])
        #     time[i] = [val / max_val for val in time[i]]
        print('Successfully generate time and Y!')
        #check invalid samples
        invalid_st = set()
        print('Invalid data size: {}'.format(len(invalid_st)))
        for i in range(len(target_users_recent_posts)):
            if len(target_users_recent_posts[i])!= post_limit or \
                len(target_users_topic_vector[i]) != post_limit:
                    invalid_st.add(i)
        if len(invalid_st):
            target_users_recent_posts = [
                target_users_recent_posts[i]
                for i in range(len(data))
                    if i not in invalid_st
            ]
            initial_post_to_recommend = [
                initial_post_to_recommend[i]
                for i in range(len(data))
                    if i not in invalid_st
            ]
            target_users_topic_vector = [
                target_users_topic_vector[i]
                for i in range(len(data))
                    if i not in invalid_st
            ]
            candidate_thread_topic_vector = [
                candidate_thread_topic_vector[i]
                for i in range(len(data))
                    if i not in invalid_st
            ]
            time = [
                time[i]
                for i in range(len(data))
                    if i not in invalid_st
            ]
            Y = [
                Y[i]
                for i in range(len(data))
                    if i not in invalid_st
            ]
        train_or_test = data['train_or_test'].values
        print('Input generated for recommendation with time successfully!')
        return target_users_recent_posts, initial_post_to_recommend, \
               target_users_topic_vector, candidate_thread_topic_vector, \
               time, Y, train_or_test
    
    
    def get_train_idxs(self, Y, train_or_test):
        pos_train_idxs = [
            i 
            for i in range(len(Y)) 
                if train_or_test[i] == self.train_name \
                    and Y[i] == 1
        ]
        neg_train_idxs = [
            i 
            for i in range(len(Y)) 
                if train_or_test[i] == self.train_name \
                    and Y[i] == 0
        ]
        neg_train_idxs = random.sample(
            neg_train_idxs,
            int(len(pos_train_idxs) * 1)
        )
        return set(pos_train_idxs + neg_train_idxs)
    

    def get_thread_to_initial_author(self, thread_to_intial_author_path):
        data = pd.read_csv(thread_to_intial_author_path)
        thread2intial_author = {}
        for i in range(len(data)):
            thread_id = int(data['thread_id'][i])
            user_id = data['user_name'][i]
            assert(thread_id not in thread2intial_author)
            thread2intial_author[thread_id] = user_id
        return thread2intial_author
    
        
    def split_recommendation_time_data(
            self,
            data,
            post_limit,
            post1_text, 
            post2_text, 
            post1_topic, 
            post2_topic, 
            Y, 
            time, 
            train_or_test
    ):
        num = len(Y)
        post1_text_train = [
            np.asarray(
                [
                    post1_text[i][j]
                    for i in range(num) 
                        if train_or_test[i] == self.train_name
                ]
            ) for j in range(post_limit)
        ] 
        post1_text_test = [
            np.asarray(
                [
                    post1_text[i][j]
                    for i in range(num) 
                        if train_or_test[i] == self.test_name
                ]
            ) for j in range(post_limit)
        ] 
        post2_text_train = np.asarray(
            [
                post2_text[i] 
                for i in range(num) 
                    if train_or_test[i] == self.train_name
            ]
        )
        post2_text_test = np.asarray(
            [
                post2_text[i] 
                for i in range(num) 
                    if train_or_test[i] == self.test_name
            ]
        )
        post1_topic_train = [
            np.asarray(
                [
                    post1_topic[i][j] 
                    for i in range(num) 
                        if train_or_test[i] == self.train_name 
                ]
            ) for j in range(post_limit)
        ] 
        post1_topic_test = [
            np.asarray(
                [
                    post1_topic[i][j] 
                    for i in range(num) 
                        if train_or_test[i] == self.test_name
                ]
            ) for j in range(post_limit)
        ] 
        post2_topic_train = np.asarray(
            [
                post2_topic[i] 
                for i in range(num) 
                    if train_or_test[i] == self.train_name
            ]
        )
        post2_topic_test = np.asarray(
            [
                post2_topic[i] 
                for i in range(num) 
                    if train_or_test[i] == self.test_name
            ]
        )
        user_thread_train = [
            [data['user_id'][i], data['thread_id'][i]]
            for i in range(num)
                if train_or_test[i] == self.train_name
        ]
        user_thread_test = [
            [data['user_id'][i], data['thread_id'][i]]
            for i in range(num)
                if train_or_test[i] == self.test_name
        ]
        train_time = [
            np.asarray(
                [
                    time[i][j]
                    for i in range(num) 
                        if train_or_test[i] == self.train_name
                ]
            ) for j in range(post_limit)
        ]
        test_time = [
            np.asarray(
                [
                    time[i][j]
                    for i in range(num) 
                        if train_or_test[i] == self.test_name
                ]
            ) for j in range(post_limit)
        ] 
        Y_test = np.asarray(
            [
                Y[i]
                for i in range(num) 
                    if train_or_test[i] == self.test_name
            ]
        )
        Y_train_input = np.asarray(
            [
                [Y[i], 1 - Y[i]]
                for i in range(num)
                    if train_or_test[i] == self.train_name
            ]
        )
            
        user_id2idx = {}
        for i in range(len(Y)):
            if data['user_id'][i] not in user_id2idx:
                user_id2idx[data['user_id'][i]] = len(user_id2idx) + 1
                
        user_ids_train = np.asarray([
            user_id2idx[data['user_id'][i]]
            for i in range(len(Y))
                if train_or_test[i] == self.train_name 
        ])
        user_ids_test = np.asarray([
            user_id2idx[data['user_id'][i]]
            for i in range(len(Y))
                if train_or_test[i] != self.train_name
        ])
        
        pos_train = sum(
            [
                Y[i] for i in range(num) 
                if train_or_test[i] == self.train_name
            ]
        )
        pos_test = sum(
            [Y[i] for i in range(num) if train_or_test[i] == self.test_name]
        )
        
        post_topic_dif_train = post_topic_dif_test = []
        time_dif_train = np.asarray([
            int(data['time_dif'][i])
            for i in range(num)
                if train_or_test[i] == self.train_name 
        ])
        time_dif_test = np.asarray([
            int(data['time_dif'][i])
            for i in range(num)
                if train_or_test[i] != self.train_name
        ])
        
        print('Positive sample number in training set: {}\n'.format(pos_train)
        + 'Negative sample number in training set: {}\n'.format(
            len(Y_train_input) - pos_train
        ) + 'Positive sample number in testing set: {}\n'.format(pos_test)
        + 'Negative sample number in testing set: {}\n'.format(
            len(Y_test) - pos_test
        ))
        print('Successfully split the dataset for recommendation with time!')
        return post1_text_train, post1_text_test, \
                post2_text_train, post2_text_test, \
                post1_topic_train, post1_topic_test, \
                post2_topic_train, post2_topic_test, \
                Y_train_input, Y_test, \
                user_thread_train, user_thread_test, \
                post_topic_dif_train, post_topic_dif_test, \
                train_time, test_time, \
                user_ids_train, user_ids_test, \
                time_dif_train, time_dif_test
               
    
    def get_train_users(self, data_folder_path, data_name):
        data = pd.read_csv(data_folder_path+data_name+'.csv')
        train_users = set([
            data['user_id'][i]
            for i in range(len(data)) 
            if data['train_or_test'][i] == self.train_name
        ])
        with open(data_folder_path + 'train_users.txt', 'w') as f:
            for item in train_users:
                f.write("%s\n" % item)
        print('Train user saved!')
        
            
    def split_data_with_x_percent_train(
            self, 
            data_folder_path, 
            data_name, 
            percent,
            suffix
    ):
        #load data
        data = pd.read_csv(data_folder_path+data_name+'{}.csv'.format(suffix))
        print('CSV file read successfully!')
        
        train_pairs_idx = [
            i for i in range(len(data)) 
            if data['train_or_test'][i] == self.train_name
        ]
        
        train_pairs_idx = set(random.sample(
            train_pairs_idx, 
            int(percent * len(train_pairs_idx))
        ))
        
        train_num = test_num = 0
        dataset = []
        for i in range(len(data)):
            if i not in train_pairs_idx:
                test_num += 1
                dataset.append([
                    data[attribute][i] \
                    if attribute != 'train_or_test' \
                    else self.test_name \
                    for attribute in self.attributes
                ])
            else:
                train_num += 1
                dataset.append([
                    data[attribute][i] \
                    if attribute != 'train_or_test' \
                    else self.train_name \
                    for attribute in self.attributes
                ])
                
        print('Train number: {}'.format(train_num))
        print('Test number: {}'.format(test_num))
                
        #save dataset
        self.save_csv(
            dataset, 
            data_folder_path + data_name + '_{}.csv'.format(
                str(percent).replace('.', '_') + suffix
            ), 
            self.attributes
        )
        print('Pecentage dataset is generated successfully!')
        
        
    def get_thread_number_by_year(self, json_data_path):
        data = self.load_data(json_data_path)
        year2threads = collections.defaultdict(set)
        for i in range(len(data)):
            if len(data[i]['posts']) < 3:
                continue
            if 'Fibromyalgia' in json_data_path or \
                'Epilepsy' in json_data_path or \
                'ALS' in json_data_path:
                year2threads[data[i]['posts'][0]['time'][6:10]].add(
                    data[i]['id']
                )
            else:
                year2threads[data[i]['posts'][0]['time'][:4]].add(
                    data[i]['id']
                )
        for key, st in year2threads.items():
            print(key, len(st))
            
    
    def get_thread_number_distribution_per_user(self, input_data_path):
        data = pd.read_csv(input_data_path)
        user2thread_num = collections.defaultdict(set)
        thread_num2user_list = collections.defaultdict(list)
        for i in range(len(data)):
            if data['train_or_test'][i] == self.test_name \
                or data['label'][i] == 0:
                continue
            user_id = data['user_id'][i]
            thread_id = data['thread_id'][i]
            user2thread_num[user_id].add(thread_id)
        for user_id, st in user2thread_num.items():
            thread_num2user_list[len(st)].append(user_id)
        for thread_num in sorted(thread_num2user_list):
            print(
                thread_num, 
                len(thread_num2user_list[thread_num]), 
                # thread_num2user_list[thread_num]
            )
            
    def get_rank_list(self, data_folder_path):
        result_data_path = data_folder_path + 'recommend_list_recall.csv'
        data = pd.read_csv(result_data_path)
        thread2rank_list = {}
        for i in range(len(data)):
            thread2rank_list[data['thread_id'][i]] = [
                'True' in s 
                for s in data['recommend_list'][i].split(',')
            ]
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
        for thread_id, rank_list in thread2rank_list.items():
            cnt5 = cnt10 = cnt30 = cnt50 = cnt100 = 0
            z5 = z10 = z30 = z50 = z100 = 0
            dcg5 = dcg10 = dcg30 = dcg50 = dcg100 = 0
            cnt = 0
            idx = 0
            while idx < len(rank_list):
                if rank_list[idx]:
                    break
                idx += 1
            if idx != len(rank_list):
                mrr.append(1 / (idx + 1))
            for i in range(len(rank_list)):
                if rank_list[i]:
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
                data_folder_path + 'thread_to_metrics.csv', 
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
        
        
        
        
        
        