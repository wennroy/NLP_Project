'''
We define the biLSTM model here
Possibly would add charCNN module as well.
PyTorch Implementation
'''
from utils import load_embedding
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class biLSTM(nn.Module):
    '''
    bi-LSTM model:
    TODO: Argument parser? There are so many variables in the constructor.
    '''
    def __init__(self, vocab, embedding_dim, embedding_dir,
                hidden_dim, hidden_fc, num_layer, num_class, dropout):
        super(biLSTM, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embedding_dir = embedding_dir
        self.hidden_dim = hidden_dim
        self.hidden_fc = hidden_fc
        self.num_layer = num_layer
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab.__len__(), 
                                            embedding_dim=embedding_dim, 
                                            padding_idx=self.vocab['<pad>'])
        self.lstm_layer = nn.LSTM(input_size=embedding_dim, 
                                  hidden_size=hidden_dim, 
                                  num_layers=num_layer,
                                  batch_first=True,
                                  bidirectional=True,
                                  dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_fc)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_fc, num_class)
        self.classifier = nn.Softmax(dim=1)
        self.copy_embedding_from_numpy()
        # self.embedding_layer.weight.requires_grad = False

    def init_states(self, batch_size):
        '''
        Initialize the hidden states in LSTM
        '''
        return (Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_dim)), 
                Variable(torch.zeros(self.num_layer*2, batch_size, self.hidden_dim)))
    
    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        pretrained = load_embedding(self.vocab, self.embedding_dir, self.embedding_dim)
        self.embedding_layer.weight = nn.Parameter(torch.tensor(pretrained,dtype=torch.float32))

    def forward(self, x, length):
#        batch_size = x.size(0)
#        sorted_length, sort_idx = torch.sort(torch.tensor(length), dim=0, descending=True)

#        _, unpack_idx = torch.sort(sort_idx, dim=0)
        #x = x[sort_idx]
#        h0, c0 = self.init_states(batch_size)
        x = self.embedding_layer(x)
        x_packed_input = pack_padded_sequence(input=x, lengths=length, batch_first=True, enforce_sorted=False)

        lstm_output, _= self.lstm_layer(x_packed_input)
        output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        out_forward = output[range(len(output)), torch.LongTensor(length) - 1, :self.hidden_dim]
        out_reverse = output[:, 0, self.hidden_dim:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        #catentate_res = torch.cat((h0[-2,:,:], h0[-1,:,:]), dim=1)

        #output = torch.index_select(output, 0, unpack_idx)
        # catentate_res = torch.index_select(catentate_res, 0, unpack_idx)
        output = self.fc1(out_reduced)
        output = self.fc2(output)
        output = self.classifier(output)
        return output

        
