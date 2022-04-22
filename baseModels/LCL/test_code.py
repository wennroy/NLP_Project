import torch
import torch.nn as nn
import argparse
import json
import numpy as np
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score
from read_data import create_data, Goemotions
from torch.utils.data import Dataset, DataLoader
from model import contextual_encoder, weighting_network

from transformers import AutoModel, AutoTokenizer
import csv
import time
import datetime
from types import SimpleNamespace


def model_eval(model, device, dev_dataloader, loss_fn, output_pred=False):
    model.eval()
    sigmoid_fn = nn.Sigmoid()
    print("predicting on dev dataset now-------")
    with torch.no_grad():
        model_result = []
        targets = []
        for dev_idx, (dev_token, dev_label, dev_attn_mask) in enumerate(dev_dataloader):
            dev_token, dev_attn_mask = dev_token.to(device), dev_attn_mask.to(device)
            model_dev_result, _ = model(dev_token, dev_attn_mask)
            model_result.extend(sigmoid_fn(model_dev_result).cpu().numpy())
            targets.extend(dev_label.cpu().numpy())

        result = calculate_metrics(np.array(model_result), np.array(targets))
        result['loss'] = loss_fn(torch.tensor(model_result), torch.tensor(targets))

    if output_pred:
        return result, model_result
    else:
        return result


# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            'accuracy': accuracy_score(y_true=target, y_pred=pred),
            'labels/f1': f1_score(y_true=target, y_pred=pred, average=None),
            'labels/precision': precision_score(y_true=target, y_pred=pred, average=None),
            'labels/recall': recall_score(y_true=target, y_pred=pred, average=None)
            }


def test(args):
    emotions_mapping = []
    with open(args.emotions_txt, "r", encoding="UTF-8") as fm:
        for line in fm:
            emotions_mapping.append(line.strip("\n"))
    print(f'Read from {args.emotions_txt}.')
    print(f'Mapping 0 to {len(emotions_mapping)-1} to label {emotions_mapping}')

    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved = torch.load(args.filepath)
        config = saved['model_config']
        config = {'num_labels': 28,
                  'dropout': args.dropout,
                  'pretrained_model': args.pretrained_model}
        config = SimpleNamespace(**config)
        model = contextual_encoder(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'dev')
        dev_dataset = Goemotions(dev_data, 28, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'dev')
        test_dataset = Goemotions(test_data, 28, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=test_dataset.collate_fn)

        sigmoid_fn = nn.Sigmoid()
        loss_fn = nn.BCELoss()
        result_dev = model_eval(model, device, dev_dataloader, loss_fn)
        result_test = model_eval(model, device, test_dataloader, loss_fn)

        for key in result_dev.keys():
            tensor_value_dev = result_dev[key]
            tensor_value_test = result_test[key]
            if key[:6] == 'labels':
                emo_dict = {}
                for i in range(len(tensor_value_dev)):  # is actually a numpy array
                    emo_dict[emotions_mapping[i]] = tensor_value_dev[i]
                result_dev[key] = emo_dict
                emo_dict = {}
                for i in range(len(tensor_value_dev)):  # is actually a numpy array
                    emo_dict[emotions_mapping[i]] = tensor_value_dev[i]
                result_test[key] = emo_dict
                continue
            result_dev[key] = float(tensor_value_dev)  # Might lose some precisions
            result_test[key] = float(tensor_value_test)
        now = datetime.datetime.now()
        current_time = now.strftime("%H-%M-%S")
        with open("dev_result"+args.dev[-3:]+current_time+args.pretrained_model.split("/")[-1]+".json", "w") as dev_file:
            json.dump(result_dev, dev_file)

        with open("test_result"+args.test[-3:]+current_time+args.pretrained_model.split("/")[-1]+".json", "w") as test_file:
            json.dump(result_test, test_file)


if __name__ == '__main__':
    from configs import get_args

    args = get_args()
    args.filepath = f'{args.batch_size}-{args.main_lrate}-{args.epoch}-{args.train[-3:]}-{args.pretrained_model.split("/")[-1]}.pt'
    # args.filepath = f'4-1e-05-5-tsv.pt'

    test(args)
