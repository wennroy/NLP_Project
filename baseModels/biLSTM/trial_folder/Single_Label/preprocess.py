'''
Plan: Use word embeddings & Bi-LSTM
This file: preprocess the data.
Credit: We adapt some of the code based on previous assignments.
'''
import csv
import numpy as np
def read_dataset(filename):
    '''
    Input: 
        filename, the directory path.
    Output:
        A list, its element is a tuple, 
        output: [t1, t2, t3, ..., tn]
        t1 = ([list of words], label in string)
        For example:
            (['Hello', 'World'], '5')
    '''
    dataset = []
    with open(filename, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            sentence, tag, _ = row
            dataset.append((sentence.lower().split(' '), tag))
    return dataset

def convert_label_vector(tag, num_class):
    '''
    Input:
        tag: string, denotes the label of the given sentence
             In GoEmotions, label could be: '8,25', '0'
        num_class: How many classes are we considering
    Output:
        label_vec: a list of length num_class,
                   with given label's entry equals to 1, o.w 0
    '''
    labels = tag.split(',')
    label = labels[0]
    return int(label)

def convert_text_to_ids(dataset, word_vocab, num_class):
    '''
    We want to represent those words by index,
    Also, the label should be converted to vector
    input:
        dataset: the dataset we read using read_dataset()
        word_vocab: a Vocab class containing those vocabularies
    Output:
        data: list of tuple elements
              the tuple consists of word_ids, label vector
    '''
    data = []
    for words, tag in dataset:
        word_ids = [word_vocab[w] for w in words]
        data.append((word_ids, convert_label_vector(tag, num_class)))
    return data

def data_iter(data, batch_size, shuffle=True):
    """
    Randomly shuffle training data, and partition into batches.
    Each mini-batch may contain sentences with different lengths.
    """
    if shuffle:
        # Shuffle training data.
        np.random.shuffle(data)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tags = [data[i * batch_size + b][1] for b in range(cur_batch_size)]
        yield sents, tags

def pad_sentences(sents, pad_id):
    """
    Adding pad_id to sentences in a mini-batch to ensure that 
    all augmented sentences in a mini-batch have the same word length.
    Args:
        sents: list(list(int)), a list of a list of word ids
        pad_id: the word id of the "<pad>" token
    Return:
        aug_sents: list(list(int)), |s_1| == |s_i|, for s_i in sents
    """
    aug_sents = sents.copy()
    lengths = []
    max_length = -1
    for sentence in aug_sents:
        max_length = max(max_length, len(sentence))
        lengths.append(len(sentence))
    # Now add <pad> to the end of each sentence
    for sentence in aug_sents:
        while len(sentence) < max_length:
            sentence.append(pad_id)
    return aug_sents, lengths
