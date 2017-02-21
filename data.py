#! /usr/bin/env python3
# coding: utf-8

import codecs
import tensorflow as tf
import numpy as np 
import os
import re
from subprocess import call

# Data loading params
flags = tf.app.flags
flags.DEFINE_integer('min_seq_len', 3, 'Minimal length of dialogs [3]')
flags.DEFINE_integer('max_seq_len', 50, 'Maximal length of dialogs [50]')
flags.DEFINE_integer('min_freq', 3, 'Keep word in vocabulary only if occurs at least N times [3]')
flags.DEFINE_string('data_dir', '/tmp/data/', 'Data directory [/tmp/data/]')
flags.DEFINE_string('url', 'http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip', 'Default URL to download corpus from, if DATA_DIR not found')
flags.DEFINE_boolean('corpus_correction', True, 'Correct erroneus lines in corpus before preprocess [True]')

FLAGS = flags.FLAGS
FLAGS._parse_flags()

 

# 1. READ DATA
def get_id2line():
    ''' 
        1. Read from 'movie-lines.txt'
        2. Create a dictionary with ( key = line_id, value = text )
    '''
    with codecs.open(os.path.join(FLAGS.data_dir, 'movie_lines.txt'), 
                     'r', encoding='utf-8', errors='ignore') as fdata:
        lines= fdata.read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
        else:
            id2line[_line[0]] = ' '
    return id2line



def get_id2genre():
    '''
        1. Read from 'movie_titles_metadata.txt'
        2. Create a dictionary with ( key = movie_id, value = [genre1, genre2, ..])
    '''
    with codecs.open(os.path.join(FLAGS.data_dir, 'movie_titles_metadata.txt'), 
                     'r', encoding='utf-8', errors='ignore') as fdata:
        lines= fdata.read().split('\n')
        
    id2genre = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 6:
            id2genre[_line[0]] = _line[-1][1:-1].replace("'","")
    return id2genre        

def get_conversations_with_movie_id():
    '''
        1. Read from 'movie_conversations.txt'
        2. Create a list of [list of line_id's]
        3. Create a list of corresponding movie_id
    '''
    conv_lines = open(os.path.join(FLAGS.data_dir, 'movie_conversations.txt'))\
        .read().split('\n')
    convs = [ ]
    movie_id = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1]\
            .replace("\\'", "'").replace("'","").replace(" ","")
        convs.append(_line.split(','))
        _line = line.split(' +++$+++ ')[2]
        movie_id.append(_line)
    return convs, movie_id

def gather_dataset_with_genres(convs, id2line, movie_id, id2genre, stride=2):
    '''
        MODIFIED!

        A1: blah
        B1: blah
        A2: blah
        B2: blah 

        are two pairs, but really two data samples for encoder and decoder 
        (Context: A1, response: B1), (Context: A1, B1, A2, response: B2). 
    '''
    contexts = []
    responses = []
    genres = []
    for conv, mid in zip(convs, movie_id):
        # in each conversation
        # 1, 2, ... i-1 lines are the context
        # ith utterance will be the response
        i = 1
        while i < len(conv):
            context = ''
            for j in range(i):
                context += ' ' + id2line[conv[j]]
            contexts.append(context)    
            responses.append(id2line[conv[i]])
            genres.append(id2genre[mid])    
            i += stride
        
    return contexts, responses, genres


def read_cornell():
    convs, movie_id = get_conversations_with_movie_id()
    id2line = get_id2line()
    id2genre = get_id2genre()
    c, r, g = gather_dataset_with_genres(convs, id2line, movie_id, id2genre)
    
    return c, r, g

# 2. TOKENIZATION
def clean_str(string):
    '''
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if FLAGS.corpus_correction:
        # Most basic way to get rid of jammed lines
        string = re.sub(r"\[", " ", string)
        string = re.sub(r"\]", " ", string)
    
    return string.strip().lower()


def tokenize_data(examples):
    '''
    Decoder data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    '''
    examples = [s.strip() for s in examples]
    
    # Split by words
    tokenized_text = [clean_str(sent) for sent in examples]
    
    return tokenized_text

def translate_single(input):
    assert(type(input) == str)
    tokenized = tokenize_data([input])
    idx = np.array(list(cont_vocab.transform(tokenized)))
    idx = np.trim_zeros(idx[0])
    return np.array([idx])

    

def main(argv=None):
    if os.path.exists(FLAGS.data_dir):
        print('0. Found DATA_DIR')
    else:
        print('0. DOWNLOAD')
        os.mkdir(FLAGS.data_dir)
        url = FLAGS.url
        print('Downloading from', url)
        print('To', './corpus.zip')
        call(['wget', url, '-O', './corpus.zip'])
        call(['unzip', '-j', 'corpus.zip', 'cornell\ movie-dialogs\ corpus/*', 
            '-d', FLAGS.data_dir])
        call(['rm', 'corpus.zip'])    
        
    print('1. READ DATA - ', end='', flush=True)
    contexts, responses, genres = read_cornell()
    print('context line count:', len(contexts))
    
    def write2file(fname, data):
        with open(os.path.join(FLAGS.data_dir, fname), 'w') as f:
            for d in data:
                f.write(d + '\n')

    write2file('context.txt', contexts)            
    write2file('response.txt', responses)
    write2file('genre.txt', genres)


    
    print('2. TOKENIZE - ', end='', flush=True)
    
    
    # Load data from files
    # data_fname = os.path.join(FLAGS.data_dir,'context.txt')
    # examples = list(open(data_fname, "r").readlines())
    
    tokenized_context = tokenize_data(contexts)
    
    lens = [len(line.split()) for line in tokenized_context]
    normlen_context = []
    normlen_genres = []
    lens = [len(line.split()) for line in tokenized_context]
    for i in range(len(tokenized_context)):
        # Dataset is very biased
        # Also guessing 24 categories on a few words is still ill proposed
        # heavy regularization
        # if 'drama' in genres[i]: continue
        
        if lens[i] > FLAGS.min_seq_len and lens[i] < FLAGS.max_seq_len:
            normlen_context.append(tokenized_context[i])
            normlen_genres.append(genres[i])
    print('longest dialog:', FLAGS.max_seq_len, 
        '- shortest dialog:', FLAGS.min_seq_len)


    print('3. VOCABULARIZE - ', end='', flush=True)
    VocProc = tf.contrib.learn.preprocessing.VocabularyProcessor
    def vocabularize(text):
        max_document_length = max([len(x.split(" ")) for x in text])
        vocab_processor = VocProc(
            max_document_length, min_frequency=FLAGS.min_freq)
        x = np.array(list(vocab_processor.fit_transform(text)))
        
        return x, vocab_processor

    global cont_vocab
    global genr_vocab    

    cont_id, cont_vocab = vocabularize(normlen_context)
    genr_id, genr_vocab = vocabularize(normlen_genres)
    print('vocabulary size:', len(cont_vocab.vocabulary_))

    print('4. CREATE GENRE LABELS')
    genr_labels = np.zeros((len(genr_id), genr_id.max()+1), dtype=int)
    for i, gid in enumerate(genr_id):
        # first row is UNK token
        genr_labels[i, gid] = True

    genr_labels = genr_labels[:, 1:]
    
    global vocabulary_size
    global num_classes
    vocabulary_size = len(cont_vocab.vocabulary_)
    num_classes = len(genr_vocab.vocabulary_) - 1 


    print('5. SAVE DATA')
    cont_vocab.save(os.path.join(FLAGS.data_dir, 'context.vocab'))
    genr_vocab.save(os.path.join(FLAGS.data_dir, 'genre.vocab'))
    np.save(os.path.join(FLAGS.data_dir, 'context'), cont_id)
    np.save(os.path.join(FLAGS.data_dir, 'genres'), genr_labels)



    # cont_id = np.load(FLAGS.data_dir + 'context.npy')
    # genr_labels = np.load(FLAGS.data_dir + 'genres.npy')



    """
        Origin:
        https://github.com/dennybritz/tf-rnn/blob/master/sequence_example.ipynb
    """

    def make_example(sequence, label):
        # The object we return
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        sequence_length = len(sequence)
        ex.context.feature['length'].int64_list.value.append(sequence_length)
        ex.context.feature['labels'].int64_list.value.extend(label)
        # This part of TF is not so verbose
        # and tutorials are rare, also serialized labels were serialized with different length

        # Reshaped a bit WildML-s tips and tricks
        # http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
        '''
        ex.feature_lists\
            .feature_list['tokens']\
            .feature.add()\
            .int64_list.value.extend(sequence)
        '''
        fl_tokens = ex.feature_lists.feature_list["tokens"]
        for token in sequence:
            fl_tokens.feature.add().int64_list.value.append(token)
        
        return ex
    
    def mat2seq(cont_id):
        print('Trimming zeros...')
        cont_list = len(cont_id) * [None]
        for i in range(len(cont_id)):
            cont_list[i] = np.trim_zeros(cont_id[i])
            if i%500 == 0: print('\r%d'%i, end='')
        return cont_list
    
    def write_TFRecord(fname, sequences, labels):
        with open(fname + '.TFRecord', 'w') as fp:
            writer = tf.python_io.TFRecordWriter(fp.name)
            print('\nSampling...')
            i = 0
            for sequence, label in zip(sequences, labels):
                
                ex = make_example(sequence, label)
                writer.write(ex.SerializeToString())
                
                if i%500 == 0: print('\r%d'%i, end='')
                i+=1
            writer.close()
            print("\nWrote to {}".format(fp.name))        


    
    
    print('7. WRITING TFRecords FILE')
    cont_list = mat2seq(cont_id)
    write_TFRecord(
        os.path.join(FLAGS.data_dir, 'cnn'), 
        cont_list, genr_labels)


    
if __name__ == '__main__':
    
    print('Running data.py')
    print('\nParameters:')
    for attr, value in sorted(FLAGS.__flags.items()):
        print('{} =\t{}'.format(attr.upper(), value))
    print('')
    
    tf.app.run()
else:    
    global cont_vocab
    global genr_vocab    


    global vocabulary_size
    global num_classes
    
    if not os.path.exists(FLAGS.data_dir):
        main()
    
    cont_vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
            os.path.join(FLAGS.data_dir, 'context.vocab'))
    vocabulary_size = len(cont_vocab.vocabulary_)

    genr_vocab = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
        os.path.join(FLAGS.data_dir, 'genre.vocab'))
    #-1 ~ UNK token
    num_classes = len(genr_vocab.vocabulary_) - 1
