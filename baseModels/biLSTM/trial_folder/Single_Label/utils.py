from __future__ import print_function
from collections import Counter
import torch
import torch.nn as nn
import numpy as np
import io
class Vocab(object):
    '''
    util #1: Vocab class
    Build vocabulary from given sentences
    '''
    def __init__(self, pad=False, unk=False, max_size=None):
        self.word2id = dict()
        self.id2word = dict()
        self.pad_id = self.unk_id = None
        if pad:
            self.add('<pad>')
            self.pad_id = self.word2id['<pad>']
        if unk:
            self.add('<unk>')
            self.unk_id = self.word2id['<unk>']
        self.max_size = max_size

    def __getitem__(self, word):
        # If a word is not in the dict, return unk_id
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        # How many words does this class hold
        return len(self.word2id)

    def __repr__(self):
        # string representation
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        # use id to retrieve word
        return self.id2word[wid]

    def add(self, word):
        # return id
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def build(self, texts):
        """
        Build vocabulary from list of sentences
        Args:
            texts: list(list(str)), list of tokenized sentences
        """
        word_freq = Counter()
        for words in texts:
            for w in words:
                word_freq[w] += 1
        
        top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)
        for word in top_k_words:
            if self.max_size:
                if len(self.word2id) < self.max_size:
                    self.add(word)
            else:
                self.add(word)

class labelMapping:
    '''
    CONST Units: store the mapping from number to string label
    '''
    def __init__(self):
        self.fine_grained = {
            '0': 'admiration',
            '1': 'amusement',
            '2': 'anger',
            '3': 'annoyance',
            '4': 'approval',
            '5': 'caring',
            '6': 'confusion',
            '7': 'curiosity',
            '8': 'desire',
            '9': 'disappointment',
            '10': 'disapproval',
            '11': 'disgust',
            '12': 'embarrassment',
            '13': 'excitement',
            '14': 'fear',
            '15': 'gratitude',
            '16': 'grief',
            '17': 'joy',
            '18': 'love',
            '19': 'nervousness',
            '20': 'optimism',
            '21': 'pride',
            '22': 'realization',
            '23': 'relief',
            '24': 'remorse', 
            '25': 'sadness',
            '26': 'surprise',
            '27': 'neutral'
        }
        self.ekman = {
            '0': 'anger',
            '1': 'disgust',
            '2': 'fear',
            '3': 'joy',
            '4': 'sadness',
            '5': 'surprise',
            '6': 'neutral'
        }
        self.emotion_groups = {
            '0': 'positive',
            '1': 'negative',
            '2': 'ambiguous',
            '3': 'neutral'
        }
    def get_mapping(self, mode='fine-grained'):
        if mode == 'fine-grained':
            return self.fine_grained
        elif mode == 'ekman':
            return self.ekman
        elif mode == 'emotion-group':
            return self.emotion_groups
        else:
            raise KeyError('No such grouping method!')

def evaluate(dataset, model, device, tag_dict=None, filename=None):
    """
    Evaluate test/dev set
    """
    predicts = []
    acc = 0
    for words, tag in dataset:
        X = torch.LongTensor([words]).to(device)
        word_len = len(words)
        scores = model(X, [word_len])
        # TODO: Need to change this "=="
        y_pred = torch.argmax(scores)
        predicts.append(y_pred)
        acc += int(y_pred == torch.tensor(tag))
    print(f'  -Accuracy: {acc/len(predicts):.4f} ({acc}/{len(predicts)})')
    if filename:
        with open(filename, 'w') as f:
            for y_pred in predicts:
                # convert tag_id to its original label
                results = (y_pred == 1).nonzero(as_tuple=True)
                tag = []
                for res in results:
                    tag.append(tag_dict[str(res)])
                prediced_labels = ','.join(tag)
                f.write(f'{prediced_labels}\n')
        print(f'  -Save predictions to {filename}')
    return acc/len(predicts)

# Load embeddings
def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    fin = io.open(emb_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if vocab.__contains__(tokens[0]):
            data[tokens[0]] = tokens[1:]
    res = np.zeros((vocab.__len__(), emb_size))
    for i, v in enumerate(vocab.word2id):
        try:
            res[i, :] = data[v]
        except KeyError:
            res[i, :] = np.zeros((1, emb_size))
            # res[i, :] = np.random.normal(scale=0.03, size = (1, emb_size))
            #res[i,:] = np.random.uniform(-1, 1, (1, emb_size))
    return res
