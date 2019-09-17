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
parser.add_argument('--learning_rate', '-lr', default=0.0001, type=float)
parser.add_argument('--data', default="sougou_business.pkl", type=str)
args = parser.parse_args()

folder = "./lr%g_%s" % (args.learning_rate, os.path.basename(args.data))
os.makedirs(folder, exist_ok=True)
DataFile = args.data


class Word:
    def __init__(self,val,tf,df):
        self.val = val
        self.tf = tf
        self.df = df
    def __repr__(self):
        pass


logger = logging.getLogger('training')
hdlr = logging.FileHandler('%s/train.log' % folder)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)
tensorboard_log_path = folder



print ("Loading file from %s." % DataFile)
sample_file = "%s/train.samples" % folder
MODEL_DUMP_DIR = folder
_, word2idx, idx2word, titles, abstracts = pickle.load(open(DataFile, 'rb'))
assert len(titles) == len(abstracts)

beg,eos,emp,unk = 0,1,2,3
learning_rate = 0.001
learning_rate = args.learning_rate

save_epoc_step = 2
dropout_keep_prob = 0.7

RESTORE = False
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

writer = tf.summary.FileWriter(tensorboard_log_path, graph=tf.get_default_graph())

def prt2file(label, x, suffix = None):
    with open(sample_file,'a') as outfile:
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

def prt(x):
    sentence = ''
    for w in x:
        if w == emp:
            continue
        sentence += idx2word[w]
    print(sentence)

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
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss,name = "op_adam_minize")
saver = tf.train.Saver(max_to_keep=100)



if args.data == "../data/basic_data_80k_v2.pkl":
    train_abstracts = abstracts[:59000]
    train_titles = titles[:59000]

    val_abstracts = abstracts[59000:60000]
    val_titles = titles[59000:60000]

    test_abstracts = abstracts[60000:]
    test_titles = titles[60000:]
elif args.data.find("business_sport") > 0:  # 84551
    train_abstracts = abstracts[:81000]
    train_titles = titles[:81000]

    val_abstracts = abstracts[81000:82000]
    val_titles = titles[81000:82000]

    test_abstracts = abstracts[82000:]
    test_titles = titles[82000:]
elif args.data.find("business") > 0:
    train_abstracts = abstracts[:54000]
    train_titles = titles[:54000]

    val_abstracts = abstracts[54000:55000]
    val_titles = titles[54000:55000]

    test_abstracts = abstracts[55000:]
    test_titles = titles[55000:]

elif args.data.find("sport") > 0:
    train_abstracts = abstracts[:78000]
    train_titles = titles[:78000]

    val_abstracts = abstracts[78000:79000]
    val_titles = titles[78000:79000]

    test_abstracts = abstracts[79000:]
    test_titles = titles[79000:]



with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    if RESTORE:
        #First let's load meta graph and restore weights
        #saver = tf.train.import_meta_graph('model/TitleGeneration-110.meta')
        saver.restore(sess,tf.train.latest_checkpoint(MODEL_DUMP_DIR))


    iter = 0
    for epoch in range(epocs):
        j = 0
        indices =[i for i in range(len(train_titles))] 
        random.shuffle(indices)
        train_titles = [train_titles[i] for i in indices]
        train_abstracts = [train_abstracts[i] for i in indices]

        while (j < len(train_titles)):
            # TODO emp uesed to train language model. 
            # the last batch 
            
            encoder_inputs_ = list(map(lambda x:rpadd(x,maxlend), train_abstracts[j : j + batch_size] ))
            decoder_inputs_ = list(map(lambda x:rpadd(x,maxlenh,prefix=beg), train_titles[j : j + batch_size]))
            decoder_targets_ = list(map(lambda x:x[1:] + [emp], decoder_inputs_))
    
            # prt(encoder_inputs_[0])
            # prt(decoder_targets_[0])
            # print("-----------------------")
            # prt(encoder_inputs_[1])
            # prt(decoder_targets_[1])
            # print("-----------------------")
            # prt(encoder_inputs_[2])
            # prt(decoder_targets_[2])
            # print("-----------------------")
            # prt(encoder_inputs_[3])
            # prt(decoder_targets_[3])
            j = j + batch_size
            

            summary, _, loss_, decoder_prediction_ = sess.run([summary_op,train_op,loss,decoder_prediction],
                feed_dict={
                    encoder_inputs : encoder_inputs_,
                    decoder_inputs : decoder_inputs_,
                    decoder_targets : decoder_targets_
            })

            print(iter, loss_)
            iter += 1
            writer.add_summary(summary, iter)





        
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
                # print(reference)
                # print(candidate)
                # print(score)
                # input("=================")
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

                if k < 10:
                    print("printing %s" % keyword)
                    prt2file("%s: [**Content**]" % keyword, test_encode_input)
                    prt2file("%s: [*Actual Headline*]" % keyword, test_decode_output)
                    prt2file("%s: [*Pred   Headline*]" % keyword, test_x, suffix="-------------------------------") 
                    
                
                BLEU_scores.append(eval_BLEU_score(pred = test_x, gt = test_decode_output))
            
            return np.mean(BLEU_scores)
                
        print("output==========================================")
        train_BLEU = qual_quant_output(train_titles, train_abstracts, "train", 100)
        val_BLEU = qual_quant_output(val_titles, val_abstracts, "val", 10000000000)        
        logger.info( "Runing in EPOC[%d] with train BLEU [%g] val BLEU [%g]" %(epoch, train_BLEU, val_BLEU))             
        
        save_path = saver.save(sess, "%s/tg-model-epoch%d"%(MODEL_DUMP_DIR, epoch))
        print("Done saving to %s" % save_path)