import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer

class contextual_encoder(nn.Module):
    def __init__(self, config):
        super(contextual_encoder, self).__init__()

        contextual = AutoModel.from_pretrained(config.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)

        self.config = config
        self.model = contextual
        self.tokenizer = tokenizer

        for params in self.model.parameters():
            params.requires_grad = True

        hid_states_output = self.model.encoder.config.hidden_size

        self.linear_ff = nn.Linear(hid_states_output, hid_states_output)
        self.pooler_dropout = nn.Dropout(config.dropout)
        self.linear2label = nn.Linear(hid_states_output, config.num_labels)
        self.pooler_af = nn.Tanh()

        nn.init.xavier_normal_(self.linear_ff.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.linear2label.weight.data, gain=1.0)

    def feedforward_output(self, cls_token):
        output = self.linear_ff(cls_token)
        output = self.pooler_dropout(output)
        output = self.pooler_af(output)
        return output

    def forward(self, input_ids, attn_mask):

        model_output = self.model(input_ids, attn_mask)
        model_output = model_output.last_hidden_state[:, 0, :]
        pooler_output = self.feedforward_output(model_output)

        feedforward_output = self.linear2label(pooler_output)
        feedforward_output = self.pooler_dropout(feedforward_output)

        normalized_output = F.normalize(model_output, dim=1)

        return feedforward_output, normalized_output

class weighting_network(nn.Module):
    def __init__(self, config):
        super(weighting_network, self).__init__()

        model = AutoModel.from_pretrained(config.pretrained_model)
        self.model = model
        hidden_size = model.encoder.config.hidden_size
        self.linear = nn.Linear(hidden_size,hidden_size)
        self.linear2label = nn.Linear(hidden_size, config.num_labels)
        self.poolerdropout = nn.Dropout(config.dropout)
        self.pooler_af = nn.Tanh()

        nn.init.xavier_normal_(self.linear2label.weight.data, gain=1.0)

    def forward(self, input_ids, attn_mask):
        model_output = self.model(input_ids, attn_mask)
        model_output = model_output.last_hidden_state[:,0,:]  # [CLS] token

        model_output = self.linear(model_output)
        model_output = self.poolerdropout(model_output)
        model_output = self.pooler_af(model_output)

        model_output = self.linear2label(model_output)
        model_output = self.poolerdropout(model_output)

        return model_output





if __name__ == "__main__":
    from types import SimpleNamespace

    config = {'num_labels': 28,
              'dropout': 0.3,
              'pretrained_model': "google/electra-base-discriminator"}
    config = SimpleNamespace(**config)

    model_main = contextual_encoder(config)
    model_helper = weighting_network(config)

    for param in model_main.parameters():
        param.requires_grad = False
    for param in model_helper.parameters():
        param.requires_grad = False

    total_params = list(model_main.parameters()) + list(model_helper.parameters())
    for param in total_params:
        param.requires_grad = True

    for param in model_main.parameters():
        print(param.requires_grad)
        break
    for param in model_helper.parameters():
        print(param.requires_grad)
        break