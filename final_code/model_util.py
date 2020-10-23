#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:12:53 2020

@author: LMD
"""

from keras.models import Model
from keras.layers import (
    Input,
    Dense, 
    Embedding, 
    LSTM, 
    Flatten, 
    Multiply,
    Activation,
    Conv2D, 
    Reshape, 
    Dropout, 
    Concatenate
)
from keras.layers.convolutional import (
    Conv1D,
    MaxPooling1D
)
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import dot
from keras import metrics
import keras.backend as K


class model_util(object):
    def __init__(self):
        self.num_conv2d_layers=1
        self.filters_2d=[16,32]
        self.kernel_size_2d=[[3,3], [3,3]]
        self.mpool_size_2d=[[2,2], [2,2]]
        self.metric_type_map = {
            'precision' : metrics.Precision(),
            'recall' : metrics.Recall(),
            'AUC' : metrics.AUC(),
            'accuracy' : metrics.Accuracy(),
        }
    
    def dot_arci(
            self,
            dense_size,
            dropout_rate,
            output_dim,
    ):
        input1 = Input(shape=(dense_size,), dtype='float32')
        input2 = Input(shape=(dense_size,), dtype='float32')
        input3 = Input(shape=(1,), dtype='float32')
        input4 = Input(shape=(1,), dtype='float32')
        
        question_dot12 = Multiply()([input3, input1])
        question_dot23 = Multiply()([input3, input2])
        action_dot12 = Multiply()([input4, input1])
        action_dot23 = Multiply()([input4, input2])
        
        question_mlp12 = Dense(dense_size, activation='relu')(question_dot12)
        question_mlp23 = Dense(dense_size, activation='relu')(question_dot23)
        action_mlp12 = Dense(dense_size, activation='relu')(action_dot12)
        action_mlp23 = Dense(dense_size, activation='relu')(action_dot23)
        
        concatenated_tensor = Concatenate(axis=1)(
            [
                question_mlp12, 
                question_mlp23, 
                action_mlp12, 
                action_mlp23
            ]
        )
        dropout = Dropout(rate=dropout_rate)(concatenated_tensor)
        relevance_qa_dense = Dense(dense_size, activation='relu')(dropout)
        output = Dense(output_dim, activation='softmax')(relevance_qa_dense)
        model = Model(
            inputs=[
                input1, 
                input2, 
                input3, 
                input4
            ], 
            outputs=output
        )
        model.compile(
            loss = 'binary_crossentropy', 
            optimizer='adam',
            metrics = [metrics.AUC()]
        )
        print('Successfully build the dot arci model!')
        return model
    
    
    def cat_arci(
            self,
            dense_size,
            dropout_rate,
            output_dim
    ):
        input1 = Input(shape=(dense_size,), dtype='float32')
        input2 = Input(shape=(dense_size,), dtype='float32')
        input3 = Input(shape=(1,), dtype='float32')
        input4 = Input(shape=(1,), dtype='float32')
        
        question_cat12 = Concatenate(axis=1)([input1, input3])
        question_cat23 = Concatenate(axis=1)([input2, input3])
        action_cat12 = Concatenate(axis=1)([input1, input4])
        action_cat23 = Concatenate(axis=1)([input2, input4])
        
        question_mlp12 = Dense(dense_size, activation='relu')(question_cat12)
        question_mlp23 = Dense(dense_size, activation='relu')(question_cat23)
        action_mlp12 = Dense(dense_size, activation='relu')(action_cat12)
        action_mlp23 = Dense(dense_size, activation='relu')(action_cat23)
        
        concatenated_tensor = Concatenate(axis=1)(
            [
                question_mlp12, 
                question_mlp23, 
                action_mlp12, 
                action_mlp23
            ]
        )
        dropout = Dropout(rate=dropout_rate)(concatenated_tensor)
        relevance_qa_dense = Dense(dense_size, activation='relu')(dropout)
        output = Dense(output_dim, activation='softmax')(relevance_qa_dense)
        model = Model(
            inputs=[
                input1, 
                input2, 
                input3, 
                input4
            ], 
            outputs=output
        )
        model.compile(
            loss = 'binary_crossentropy', 
            optimizer='adam',
            metrics = ['accuracy']
        )
        print('Successfully build the cat arci model!')
        return model
    
    
    def get_CNN(
            self,
            text_input,
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size
    ):
        embedding = Embedding(
            max_features, 
            embed_dim, 
            input_length=maxlen
        )(text_input)
        conv = Conv1D(
            filters=filter_num, 
            kernel_size=kernel_size, 
            padding='same', 
            activation='relu'
        )(embedding)
        maxpool = MaxPooling1D(pool_size=pool_size)(conv)
        return maxpool
    
    
    def get_arci(
            self,
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_size,
            output_dim
    ):
        input1 = Input(shape=(maxlen,), dtype='int32')
        input2 = Input(shape=(maxlen,), dtype='int32')
        
        maxpool1 = self.get_CNN(
            input1, 
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size
        )
        maxpool2 = self.get_CNN(
            input2,
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size
        )
        
        comb_layer = Concatenate(axis=1)([maxpool1, maxpool2])
        
        flatten = Flatten()(comb_layer)
        dropout = Dropout(dropout_rate)(flatten)
        
        mlp=Dense(dense_size, activation='relu')(dropout)
        output = Dense(output_dim, activation='softmax')(mlp)
        
        model = Model(inputs=[input1, input2], outputs=output)
        model.compile(
            loss = 'binary_crossentropy', 
            optimizer='adam',
            metrics = ['accuracy']
        )
        
        print('Successfully generate ARC-I model!')
        return model
    
    
    def get_match_pyramid(
            self,
            maxlen,
            max_features,
            embed_dim,
            dropout_rate
    ):
        input1 = Input(shape=(maxlen,), dtype='int32')
        input2 = Input(shape=(maxlen,), dtype='int32')
        
        embedding1 = Embedding(
            max_features, 
            embed_dim, 
            input_length=maxlen, 
            dropout=dropout_rate
        )(input1)
        embedding2 = Embedding(
            max_features, 
            embed_dim, 
            input_length=maxlen, 
            dropout=dropout_rate
        )(input2)
        
        layer1_dot=dot([embedding1, embedding2], axes=-1)
        layer1_dot=Reshape((maxlen, maxlen, -1))(layer1_dot)
            
        layer1_conv=Conv2D(
            filters=8, 
            kernel_size=5, 
            padding='same'
        )(layer1_dot)
        layer1_activation=Activation('relu')(layer1_conv)
        z=MaxPooling2D(pool_size=(2,2))(layer1_activation)
            
        for i in range(self.num_conv2d_layers):
            z=Conv2D(
                filters=self.filters_2d[i], 
                kernel_size=self.kernel_size_2d[i], 
                padding='same'
            )(z)
            z=Activation('relu')(z)
            z=MaxPooling2D(
                pool_size=(
                    self.mpool_size_2d[i][0], 
                    self.mpool_size_2d[i][1]
                )
            )(z)
                
        pool1_flat=Flatten()(z)
        pool1_flat_drop=Dropout(rate=dropout_rate)(pool1_flat)
        mlp1=Dense(32)(pool1_flat_drop)
        mlp1=Activation('relu')(mlp1)
        out=Dense(2, activation='softmax')(mlp1)
        
        model=Model(inputs=[input1, input2], outputs=out)
        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        print('Successfully generate MatchPyramid model!')
        return model
    
    
    def baseline_predict(self, test12, test23, test_question, test_action):
        y_p = [
            test12[i][0] * \
            test23[i][0] * \
            (test_question[i] + test_action[i]) / 2
            for i in range(len(test12))
        ]
        y_p = [
            [val, 1.0 - val]
            for val in y_p
        ]
        print('Finish prediction using baseline model!')
        return y_p
    
    
    def remove_softmax_layer(self, relevance_model):
        relevance_model.layers.pop()
        relevance_model = Model(
            inputs=relevance_model.inputs, 
            outputs=relevance_model.layers[-1].output
        )
        return relevance_model
    
    
    def get_CLIR(
            self,
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
    ):
        input1 = Input(shape=(maxlen,), dtype='int32')
        input2 = Input(shape=(maxlen,), dtype='int32')
        input3 = Input(shape=(topic_dim,), dtype='float')
        input4 = Input(shape=(topic_dim,), dtype='float')
        if add_user_id:
            input5 = Input(shape=(1,), dtype='float')
        if add_time_dif:
            input6 = Input(shape=(1,), dtype='float')
        
        maxpool1 = self.get_CNN(
            input1,
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size
        )
        maxpool2 = self.get_CNN(
            input2,
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size
        )
        
        flatten1 = Flatten()(maxpool1)
        flatten2 = Flatten()(maxpool2)
        
        comb1 = Concatenate(axis=1)([flatten1, input3])
        comb2 = Concatenate(axis=1)([flatten2, input4])
        
        mlp1 = Dense(dense_size, activation='relu')(comb1)
        mlp2 = Dense(dense_size, activation='relu')(comb2)
        
        dropout1 = Dropout(dropout_rate)(mlp1)
        dropout2 = Dropout(dropout_rate)(mlp2)
        
        if add_user_id and add_time_dif:
            comb_layer = Concatenate(axis=1)(
                [dropout1, dropout2, input5, input6]
            )
        elif add_time_dif:
            comb_layer = Concatenate(axis=1)([dropout1, dropout2, input6])
        elif add_user_id:
            comb_layer = Concatenate(axis=1)([dropout1, dropout2, input5])
        else:
            comb_layer = Concatenate(axis=1)([dropout1, dropout2])
        
        mlp = Dense(dense_size, activation='relu')(comb_layer)
        output = Dense(output_dim, activation='softmax')(mlp)
        
        input_data = [input1, input2, input3, input4]
        
        if add_user_id:
            input_data += [input5]
        if add_time_dif:
            input_data += [input6]
        
        model = Model(inputs=input_data, outputs=output)
        
        def custom_loss(layer1, layer2):
            def loss(y_true,y_pred):
                return K.mean(
                    K.square(y_pred - y_true)) \
                    + K.mean(K.square(layer1)) \
                    + K.mean(K.square(layer2)
                )
            return loss
        
        model.compile(
            loss = custom_loss(dropout1, dropout2), \
            optimizer='adam', \
            metrics = [self.metric_type_map[metric_type]]
        )
        print('Successfully generated CLIR model')
        return model
    
    
    def get_CLLIR(
            self,
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
    ):
        post1_text = [
            Input(shape=(maxlen,), dtype='int32') 
            for _ in range(post_limit)
        ]
        post2_text = Input(shape=(maxlen,), dtype='int32')
        post1_topic = [
            Input(shape=(topic_dim,), dtype='float')
            for _ in range(post_limit)
        ]
        post2_topic = Input(shape=(topic_dim,), dtype='float')
        time = [
            Input(shape=(time_dim,), dtype='float')
            for _ in range(post_limit)
        ]
        if add_user_id:
            user_id = Input(shape=(1,), dtype='float')
        if add_time_dif:
            time_dif = Input(shape=(1,), dtype='float')
            
        #CNN
        post1_text_max_pooling = [
            self.get_CNN(
                v,
                max_features,
                embed_dim,
                maxlen,
                filter_num,
                kernel_size,
                pool_size
            )
            for v in post1_text
        ]
        post2_text_max_pooling = self.get_CNN(
            post2_text,
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size
        )
        #Flattern
        flatten1 = [
            Flatten()(v)
            for v in post1_text_max_pooling
        ]
        flatten2 = Flatten()(post2_text_max_pooling)
        # print('flatten1', flatten1[0].shape)
        # print('flatten2', flatten2.shape)
        comb1 = [
            Concatenate(axis=1)([flatten1[i], post1_topic[i]])
            for i in range(post_limit)
        ]
        comb2 = Concatenate(axis=1)([flatten2, post2_topic])
        # print('comb1', comb1)
        mlp1 = [
            Dense(dense_size, activation='relu')(v)
            for v in comb1
        ]
        # print('mlp1', mlp1)
        mlp2 = Dense(dense_size, activation='relu')(comb2)
        # print('time', time[0].shape)
        dropout1 = Concatenate(axis=1)(
            [
                Concatenate(axis=1)(
                    [
                        Dropout(dropout_rate)(mlp1[i]),
                        time[i]
                    ]
                )
                for i in range(post_limit)
            ]
        )
        
        reshape1 = Reshape((post_limit, dense_size+time_dim))(dropout1)
        
        dropout2 = Dropout(dropout_rate)(mlp2)
        
        time_seq = LSTM(
            lstm_out, 
            dropout_U=dropout_U, 
            dropout_W=dropout_W
        )(reshape1)
        
        if add_user_id and add_time_dif:
            comb_layer = Concatenate(axis=1)(
                [time_seq, dropout2, user_id, time_dif]
            )
        elif add_time_dif:
            comb_layer = Concatenate(axis=1)([time_seq, dropout2, time_dif])
        elif add_user_id:
            comb_layer = Concatenate(axis=1)([time_seq, dropout2, user_id])
        else:
            comb_layer = Concatenate(axis=1)([time_seq, dropout2])
        
        mlp = Dense(dense_size, activation='relu')(comb_layer)
        output = Dense(2, activation='softmax')(mlp)
        
        input_data = post1_text \
        + [post2_text] \
        + post1_topic \
        + [post2_topic] \
        + time 
        
        if add_user_id:
            input_data += [user_id]
        if add_time_dif:
            input_data += [time_dif]
        
        model = Model(inputs=input_data, outputs=output)
        
        def custom_loss(layer1, layer2):
            def loss(y_true,y_pred):
                return K.mean(K.square(y_pred - y_true)) \
                    + K.mean(K.square(layer1)) \
                    + K.mean(K.square(layer2))
            return loss
        
        model.compile(
            loss = custom_loss(dropout1, dropout2), 
            optimizer='adam',
            metrics = [self.metric_type_map[metric_type]]
        )
        print('Successfully generated CLLIR model')
        return model