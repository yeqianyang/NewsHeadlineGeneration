
import json 

import pickle
import numpy as np
import random
import jieba 
import multiprocessing
import re
import glob

word2idx, idx2word ,allwords, corpus =  None, None,{},[]
DUMP_FILE = './sougou.pkl'
check_sample_size = 10
TF_THRES = 5
DF_THRES = 2


id_beg = 0
id_eos = 1
id_emp = 2
id_unk = 3

r = None

class Word:
    def __init__(self,val,tf,df):
        self.val = val
        self.tf = tf
        self.df = df
    def __repr__(self):
        pass

def read_file():
    list_name = glob.glob('./data/newsdata/newsdata/news*.txt')
    
    out = []
    visited_titles = {}

    for filename in list_name:
        file = open(filename, 'rb').read().decode("utf8")

        patternURL = re.compile(r'<url>(.*?)</url>', re.S)
        patternCtt = re.compile(r'<content>(.*?)</content>', re.S)
        patternTitle = re.compile(r'<contenttitle>(.*?)</contenttitle>', re.S)
        classes = patternURL.findall(file)
        contents = patternCtt.findall(file)
        titles = patternTitle.findall(file)

        
        for content, title, cat in zip(contents, titles, classes):
            if cat.find('business') > 0 or cat.find("sport") > 0:
                if len(title) < 8 or len(content) < 20:
                    continue
                if title not in visited_titles:
                    out.append({ "content": content, "title": title })
                    visited_titles[title] = 1
                            
    return out

def parse_all_crawled_data():
    res = []
    
    all_data = read_file()
    
    



    for data in all_data:
        # data = json.loads(data)
        # key = data.get("group_id")
        key = "whatever"
        title = data["title"].replace('\t',' ')
        abstract = data["content"].replace('\t',' ')
        if abstract == "":
            continue
        res.append((key,title,abstract))
    return res    

def cal_word_tf_df(corpus):
    words = {}
    title_abstract_pairs = []
    for doc in corpus:    
        title, abstract = doc[1].lower(),doc[2].lower()
        ts_ = list(jieba.cut(title,cut_all = False))
        as_ = list(jieba.cut(abstract,cut_all = False))
        title_abstract_pairs.append((ts_, as_))
        # acumulate the term frequency
        for word in ts_ + as_:
            if not words.get(word):
                words[word] = Word(val = word,tf = 1,df = 0)
            else:
                words[word].tf += 1
        # acummulate the doc frequency
        for word in set(ts_ + as_):
            words[word].df += 1
    return words,title_abstract_pairs

def build_idx_for_words_tf_df(chars,tf_thres = TF_THRES, df_thres = DF_THRES):
    
    start_idx = id_unk + 1

    char2idx = {}
    idx2char = {}

    char2idx['<eos>'] = id_eos
    char2idx['<unk>'] = id_unk
    char2idx['<emp>'] = id_emp
    char2idx['<beg>'] = id_beg
    #filter out tf>20 and df > 10 terms
    chars = filter(lambda char:char.tf > tf_thres or char.df > df_thres,chars)
    char2idx.update(dict([(char.val,start_idx + idx) for idx,char in enumerate(chars)]))
    idx2char = dict([(idx,char) for char,idx in char2idx.items()])
    return char2idx, idx2char

def prt(label, x):
    print (label+':',)
    for w in x:
        if w == id_emp:
            continue
        print (idx2word[w])

# def worker():
    
#     corpus = parse_all_crawled_data()
#     print ("get docs :[%d]!"%(i,len(corpus)))
#     words,sub_corpus = cal_word_tf_df(corpus)
#     return words,sub_corpus

def combine_results():
    global corpus,word2idx,idx2word, allwords

    all_data = parse_all_crawled_data()
    print ("get docs :[%d]!"%(len(all_data)))
    words,corpus = cal_word_tf_df(all_data)

    
    for word in words:
        if word not in allwords:
            allwords[word] = Word(val = word,tf = 0,df = 0)
        allwords[word].tf += words[word].tf
        allwords[word].df += words[word].df
    word2idx, idx2word = build_idx_for_words_tf_df(allwords.values())

def dump_all_results():
    datafile = open(DUMP_FILE,'wb')
    titles, abstracts = [],[]
    for ts_,as_ in corpus:
        titles.append([word2idx.get(word,id_unk) for word in ts_])
        abstracts.append([word2idx.get(word,id_unk) for word in as_])
    pickle.dump((allwords, word2idx, idx2word, titles, abstracts),datafile,-1)

def check_dump():
    allwords, word2idx, idx2word, titles, abstracts = pickle.load(open(DUMP_FILE, "rb"))
    print( "allwords size is:",len(allwords))
    print( "word2idx size is:",len(word2idx))
    print( "titles size is:",len(titles))
    
    length_title = []
    length_abstracts = []
    for title, abstract in zip(titles, abstracts):
        length_title.append(len(title))
        length_abstracts.append(len(abstract))
    print("average length title:", np.mean(length_title))
    print("average length abstract:", np.mean(length_abstracts))
    
        
        

worker_size = 10

combine_results()
dump_all_results()
check_dump()
print( "all job finished!"  )
