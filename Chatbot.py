# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:49:08 2020

@author: Lenovo
"""

import numpy as np
import re
import tensorflow as tf


###### Data Preprocessing #######

#import dataset

lines = open('movie_lines.txt',encoding ='utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt',encoding ='utf-8', errors = 'ignore').read().split('\n')

# Creating A dictionary that maps each line  and its id

id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]=_line[4]
        
# Creating The List Of Conversations 

conversation_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(','))

# Getting Separately the questions and answers
questions = []
answers = []
for conversation in conversation_ids : 
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'ve"," have",text)
    text = re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","would not",text)
    text = re.sub(r"can't","can not",text)
    text = re.sub(r"[-()\"#?/@;:<>{}~|.,]","",text)
    return text

#cleaning Questions

clean_questions = []

for question in questions:
    clean_questions.append(clean_text(question))
#cleaning answers

clean_answers = []

for answer in answers:
    clean_answers.append(clean_text(answer))
    
## creating a dict that maps each word to its number of ocurrences

word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word] +=1
 
#Creating to dict. that maps the question word and answer word to unique integers
threshold =20
questionwords2init = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionwords2init[word] = word_number
        word_number +=1
answerswords2init ={}
for word, count in word2count.items():
    if count >= threshold:
        answerswords2init[word] = word_number
        word_number +=1 

# add the last tokens to these dictionary

tokens =['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionwords2init[token] = len(questionwords2init) + 1
for token in tokens:
    answerswords2init[token] = len(answerswords2init) + 1

# Creating the Inverse Dictionary 
 
answersint2word = {w_i : w for w,w_i in answerswords2init.items() }

# adding EOS in the end of list
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
# Translating all the Question and Answer into integers
# replacing all the words that were filtered out by <OUT>

question_into_int =[]
for question in clean_questions:
     ints=[]
     for word in question.split():
         if word not in questionwords2init:
             ints.append(questionwords2init['<OUT>'])
         else:
            ints.append(questionwords2init[word])
     question_into_int.append(ints)
    
answer_into_int =[]
for answer in clean_answers:    
    ints=[]    
    for word in answer.split():
         if word not in answerswords2init:
             ints.append(answerswords2init['<OUT>'])
         else:
            ints.append(answerswords2init[word])
    answer_into_int.append(ints)         

# sorting answer and Question into by the length of questions

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25+1):
    for i in enumerate(question_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(question_into_int[i[0]])
            sorted_clean_answers.append(question_into_int[i[0]])


##################  BULIDING SEQ2SEQ MODEL ####################

### Creating placeholders for inputs and the targets
def model_input():
    inputs = tf.placeholder(tf.int32,[None , None],name = 'input')
    targets = tf.placeholder(tf.int32, [None, None],name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs , targets, lr,keep_prob

# preprocessing the targets

def preprocess_target(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1],[1,1])
    preprocessed_targets = tf.concat([left_side,right_side],1)
    return preprocessed_targets
 
## Creating Encoder RNN Layer

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob= keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout]* num_layers)
    encoder_output , encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw= encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs, dtype=tf.float32)
    return encoder_state

#DECODING TRAINING SET
   
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope,output_function, keep_prob,batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', 
                                                                                                                                    num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_values,
                                                                              attention_keys,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decode(decoder_cell, 
                                                                                                             training_decoder_function,
                                                                                                             decoder_embedded_input,
                                                                                                             sequence_length,
                                                                                                             scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)

# decode test/validation set

def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id , eos_id, maximum_length, num_words, sequence_length, decoding_scope,output_function, keep_prob,batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', 
                                                                                                                                    num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                                encoder_state[0],
                                                                              attention_values,
                                                                              attention_keys,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix, 
                                                                              sos_id ,
                                                                              eos_id,
                                                                              maximum_length, 
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decode(decoder_cell, 
                                                                                                              test_decoder_function,
                                                                                                              scope = decoding_scope)
    return test_predictions

# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn_layer(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_target(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


# Setting hyperparameters

epochs = 100 
batch_size= 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.001
learning_rate_decay = 0.9
minimum_learning_rate = 0.0001
keep_probability = 0.5

# Defining Session

tf.reset_default_graph()
session=tf.InteractiveSession()

# Loading Model Inputs

inputs , targets,lr, keep_prob  = model_input()

#Setting the Sequence Length 
sequence_length = tf.placeholder_with_default(25, None , name = 'sequence_length')

# getting the shape of input Tensor

input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2init),
                                                       len(questionwords2init),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionwords2init)

#Setting up the Loss error, optimizer , Gradient Clipping 

with tf .name_scope("Optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients  = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.,5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)