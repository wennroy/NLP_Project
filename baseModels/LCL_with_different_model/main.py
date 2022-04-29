import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score

import random
import numpy as np
import os
import json
import argparse

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer

import csv
import time
from types import SimpleNamespace
from model import contextual_encoder, weighting_network
from read_data import create_data, Goemotions
from configs import get_args
from test_code import model_eval, calculate_metrics, test
from loss import LCL


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train():
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}")

    train_data, num_labels = create_data(args.train, 'train')
    train_dataset = Goemotions(train_data, num_labels, args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)

    dev_data = create_data(args.dev, 'dev')
    dev_dataset = Goemotions(dev_data, num_labels, args)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    config = {'num_labels': num_labels,
              'dropout': args.dropout,
              'pretrained_model': args.pretrained_model,
              'weighting_model': args.weighting_model}
    config = SimpleNamespace(**config)

    model_main = contextual_encoder(config)
    model_helper = weighting_network(config)
    model_main, model_helper = model_main.to(device), model_helper.to(device)

    total_params = list(model_main.named_parameters()) + list(model_helper.named_parameters())
    num_training_steps = int(len(train_data) * args.epoch)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in total_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.decay},
        {'params': [p for n, p in total_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.main_lrate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for name, param in total_params:
        param.requires_grad = True

    start_time = time.time()
    loss_fn = {'CE': nn.BCELoss(), 'LCL': LCL(temperature=args.temperature), 'lambda_loss': args.lambda_loss}
    sigmoid_fn = nn.Sigmoid()

    eval_iter = 1500
    batch_iter = 500
    best_dev_macf1 = 0
    training_loss_result = []
    EARLY_STOP_COUNT = 0
    EARLY_STOP = 10
    for epoch in range(args.epoch):
        batch_losses = []
        for batch_idx, (
                batch_token, batch_attn_mask, batch_weighting_token, batch_weighting_attn_mask,
                batch_label) in enumerate(train_dataloader):
            model_main.train()
            model_helper.train()
            optimizer.zero_grad()
            batch_token, batch_label, batch_attn_mask = batch_token.to(device), batch_label.to(
                device), batch_attn_mask.to(device)
            batch_weighting_token, batch_weighting_attn_mask = batch_weighting_token.to(
                device), batch_weighting_attn_mask.to(device)
            classifier_output, normalized_output = model_main(batch_token, batch_attn_mask)
            weighting_output = model_helper(batch_weighting_token, batch_weighting_attn_mask)

            lcl_loss = (loss_fn["lambda_loss"] * loss_fn["CE"](sigmoid_fn(classifier_output), batch_label)) + (
                        (1 - loss_fn["lambda_loss"]) * loss_fn["LCL"](normalized_output, batch_label, weighting_output))
            ce_loss = loss_fn["lambda_loss"] * loss_fn["CE"](sigmoid_fn(weighting_output), batch_label)

            loss = lcl_loss + ce_loss
            batch_loss_value = loss.item()
            loss.backward()
            optimizer.step()
            model_main.zero_grad()
            model_helper.zero_grad()
            lr_scheduler.step()
            optimizer.zero_grad()

            batch_losses.append(batch_loss_value)
            training_loss_result.append(batch_loss_value)

            if batch_idx % batch_iter == 0:
                avg_batch_loss = torch.mean(torch.tensor(batch_losses).type(torch.float32))
                print(
                    f'Current {batch_idx} iterations ({args.batch_size} batch size) trained. Last {batch_iter} iter has {avg_batch_loss} loss. Time elapsed: {time.time() - start_time}s')
                batch_losses = []

            if batch_idx % eval_iter == 0:
                EARLY_STOP_COUNT += 1
                result = model_eval(model_main, device, dev_dataloader, loss_fn["CE"])
                print("epoch:{:2d} iter:{:d} dev: "
                      "micro f1: {:.3f} "
                      "macro f1: {:.3f} "
                      "samples f1: {:.3f} "
                      "sample precision: {:.3f} "
                      "accuracy: {:.3f} "
                      "loss: {:.4f} "
                      "time elapsed: {:.2f}".format(epoch, batch_idx,
                                                    result['micro/f1'],
                                                    result['macro/f1'],
                                                    result['samples/f1'],
                                                    result['samples/precision'],
                                                    result['accuracy'],
                                                    result['loss'],
                                                    time.time() - start_time))
                if best_dev_macf1 < result['macro/f1']:
                    print(f'Beat the best dev macro f1 score {best_dev_macf1}, current best is {result["macro/f1"]}')
                    EARLY_STOP_COUNT = 0
                    best_dev_macf1 = result['macro/f1']
                    save_model(model_main, optimizer, args, config, args.filepath)

                if EARLY_STOP_COUNT > EARLY_STOP:
                    print(f'EARLY STOP!! Dev accuracy hasn\'t increased for {EARLY_STOP} EPOCHS!')
                    break
        EARLY_STOP_COUNT += 1
        result = model_eval(model_main, device, dev_dataloader, loss_fn["CE"])
        print("epoch:{:2d} dev: "
              "micro f1: {:.3f} "
              "macro f1: {:.3f} "
              "samples f1: {:.3f} "
              "sample precision: {:.3f} "
              "accuracy: {:.3f} "
              "loss: {:.4f} "
              "time elapsed: {:.2f}".format(epoch,
                                            result['micro/f1'],
                                            result['macro/f1'],
                                            result['samples/f1'],
                                            result['samples/precision'],
                                            result['accuracy'],
                                            result['loss'],
                                            time.time() - start_time))

        if best_dev_macf1 < result['macro/f1']:
            print(f'Beat the best dev macro f1 score {best_dev_macf1}, current best is {result["macro/f1"]}')
            EARLY_STOP_COUNT = 0
            best_dev_macf1 = result['macro/f1']
            save_model(model_main, optimizer, args, config, args.filepath)
        if EARLY_STOP_COUNT > EARLY_STOP:
            print(f'EARLY STOP!! Dev accuracy hasn\'t increased for {EARLY_STOP} EPOCHS!')
            break


def save_model(model, optimizer, args, config, filepath):
    print("Saving the model...")
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.batch_size}-{args.main_lrate}-{args.epoch}-{args.train[-3:]}-{args.pretrained_model.split("/")[-1]}.pt'
    print(args.filepath)
    train()
    test(args)
