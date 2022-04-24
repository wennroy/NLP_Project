import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


class Basic_BERTweet_Classifier(nn.Module):

    def __init__(self, config):
        super(Basic_BERTweet_Classifier, self).__init__()
        bertweet = AutoModel.from_pretrained(config.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

        self.config = config
        self.Bertweet = bertweet
        self.tokenizer = tokenizer

        for params in self.Bertweet.parameters():
            params.requires_grad = True

        hid_states_output = bertweet.pooler.dense.out_features
        self.linear1 = nn.Linear(in_features=hid_states_output, out_features=hid_states_output)
        self.linear2 = nn.Linear(in_features=hid_states_output, out_features=config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.tanh_af = nn.Tanh()
        self.sigm = nn.Sigmoid()

        nn.init.xavier_normal_(self.linear1.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.linear2.weight.data, gain=1.0)

    def forward(self, x, attn_mask):
        input_ids = x
        Bertweet_output = self.Bertweet(input_ids, attn_mask)
        pooler_output = Bertweet_output["pooler_output"]

        feedfoward_output = self.linear1(pooler_output)
        feedfoward_output = self.dropout(feedfoward_output)
        feedfoward_output = self.tanh_af(feedfoward_output)
        feedfoward_output = self.linear2(feedfoward_output)
        output = self.sigm(feedfoward_output)

        return output

