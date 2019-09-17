# coding:utf-8
import tensorflow as tf
import sys
import logging
import pickle
import random
import os
from tensorflow.contrib import rnn
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', default="sougou_business_sport.pkl", type=str)
parser.add_argument('--model', default="./lr0.001_sougou_business_sport.pkl/tg-model-epoch66", type=str)
args = parser.parse_args()

DataFile = args.data


class Word:
    def __init__(self,val,tf,df):
        self.val = val
        self.tf = tf
        self.df = df
    def __repr__(self):
        pass




print ("Loading file from %s." % DataFile)
all_words, word2idx, idx2word, titles, abstracts = pickle.load(open(DataFile, 'rb'))
assert len(titles) == len(abstracts)

beg,eos,emp,unk = 0,1,2,3

save_epoc_step = 2
dropout_keep_prob = 0.7


batch_size = 128
epocs = 1500


maxlena=100 # 0 - if we dont want to use description at all
maxlent=20
maxlen = maxlena + maxlent
maxlenh = maxlent
maxlend = maxlena

vocab_size = len(word2idx)
embedding_size = 100
memory_dim = 512

# for cnn encoder use
filter_sizes = [1,2,3,4,5,6,8,10]
num_filters = 64

# for rnn deocoder use ,GRU cell memory size. same as encoder state


encoder_inputs = tf.placeholder(tf.int32, shape=[None,maxlend], name='encoder_inputs')
decoder_targets = tf.placeholder(tf.int32,shape=(None, maxlenh), name='decoder_targets')
decoder_inputs = tf.placeholder(tf.int32, [None, maxlenh], name = "decoder_inputs")

embeddings = tf.Variable(
            tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="embeddings")


def prt2file(label, x, suffix = None):
    with open('test_output.txt','a') as outfile:
        outfile.write((label+':')),
        for w in x:
            if w == emp:
                continue
            outfile.write(idx2word[w]),
        outfile.write("\n")
        outfile.flush()

        if suffix:
            outfile.write("%s\n" % suffix)
            outfile.flush()

# cnn as encode
def CNNEncoder(encoder_inputs):
    #train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    # to expand one dim for CNN
    embed_expanded = tf.expand_dims(encoder_inputs_embedded,-1)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embed_expanded,
                W,  
                strides=[1, 1, 1, 1], 
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            #print h.shape
            # Max-pooling over the outputs
            pooled = tf.nn.max_pool(
                h,  
                ksize=[1, maxlend - filter_size + 1, 1, 1], 
                strides=[1, 1, 1, 1], 
                padding='VALID',
                name="pool")          
            pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs,3)
    #print h_pool.shape
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    #print h_pool_flat.shape

    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob,name="dropout")
    return h_drop

def RNNDecoder(encoder_state,decoder_inputs):
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
    #from tensorflow.models.rnn import rnn_cell, seq2seq
    cell = rnn.GRUCell(memory_dim)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
        cell, decoder_inputs_embedded,
        initial_state=encoder_state,
        dtype=tf.float32,scope="plain_decoder1")
    return decoder_outputs, decoder_final_state 

def rpadd(x, maxlen=maxlenh, eos=eos,prefix=None):
    assert maxlen >= 0
    
    if prefix != None:
        x = [prefix] + x
    n = len(x)
    if n > maxlen - 1 :
        x = x[:maxlen - 1]
        n = maxlen - 1
    res = x + [eos] + [emp] * (maxlen - n - 1) 
    assert len(res) == maxlen
    return res

def prepare_sentences():
    sents = []
    segs = map(lambda x:x,['。','？','！','；'])
    splits = set(map(lambda x:word2idx[x],segs))
    for abstract in abstracts:
        i,start_idx = 0 ,0
        while(i < len(abstract)):
            if abstract[i] in splits:
                sents.append(abstract[start_idx:i+1])
                start_idx = i + 1
            i += 1
    return titles + sents



encoder_state = CNNEncoder(encoder_inputs)
decoder_outputs, _ = RNNDecoder(encoder_state,decoder_inputs)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
labels = tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels = labels,
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy,name = "loss")
tf.summary.scalar("cost", loss)
summary_op = tf.summary.merge_all()

decoder_prediction = tf.argmax(decoder_logits, 2,name = "decoder_prediction")
saver = tf.train.Saver(max_to_keep=100)


if args.data.find("business_sport") > 0:  # 84551
    test_abstracts = abstracts[82000:]
    test_titles = titles[82000:]

    val_abstracts = abstracts[81000:82000]
    val_titles = titles[81000:82000]



with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    saver.restore(sess,args.model)


    iter = 0
                  
    def qual_quant_output(titles, abstracts, keyword, n):        
        def eval_BLEU_score(pred, gt):
            parsed_pred = []
            parsed_gt = []
            for w in pred:
                if w == emp:
                    continue
                parsed_pred.append(idx2word[w])
            for w in gt:
                if w == emp:
                    continue
                parsed_gt.append(idx2word[w])
                
            reference = [parsed_gt]
            candidate = parsed_pred
            score = sentence_bleu(reference, candidate)
            return score

        BLEU_scores = []
        for k in range(min(n, len(titles))):
            test_encode_input = rpadd(abstracts[k], maxlend)
            test_decode_output = rpadd(titles[k], maxlenh)
            
            test_x = []
            for l in range(maxlenh):
                new_decoder_input = rpadd(test_x, maxlenh, prefix=beg)
                decoder_prediction_ = sess.run([decoder_prediction],
                            feed_dict = {
                            encoder_inputs : [test_encode_input],
                            decoder_inputs : [new_decoder_input]
                            }
                )
                test_x.append(decoder_prediction_[0][0][l])
                if decoder_prediction_[0][0][l] == eos:
                    break

            
            print("printing %s" % keyword)
            prt2file("%s: [**Content**]" % keyword, test_encode_input)
            prt2file("%s: [*Actual Headline*]" % keyword, test_decode_output)
            prt2file("%s: [*Pred   Headline*]" % keyword, test_x, suffix="-------------------------------") 
            
            
            BLEU_scores.append(eval_BLEU_score(pred = test_x, gt = test_decode_output))
        
        return np.mean(BLEU_scores)

    val_BLEU = qual_quant_output(val_titles, val_abstracts, "val", 10000000000)
    test_BLEU = qual_quant_output(test_titles, test_abstracts, "test", 10000000000)