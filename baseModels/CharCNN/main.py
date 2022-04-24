'''
This file helps us train charCNN models in a modularized fashion
'''
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil
from data_utils import MyDataset
from torch.utils.tensorboard import SummaryWriter
from model import CharacterLevelCNN
from eval import get_evaluation

def get_args():
    '''
    When you train the model from your terminal,
    argument parser is needed.
    '''
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Character-level convolutional networks for text classification""")
    # The alphabet considered in the original paper.
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    # the max sequence length
    # In one of our experiments, we truncate it to 705
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    # The paper has two sizes of 'CONV'
    # We did not observe significant improvement on GoEmotions when using the larger one.
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    # Optimizer
    parser.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-n", "--num_epochs", type=int, default=20)
    parser.add_argument("-l", "--lr", type=float, default=0.001)  # recommended learning rate for sgd is 0.01, while for adam is 0.001
    parser.add_argument("-y", "--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("-w", "--es_patience", type=int, default=3,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    parser.add_argument("-v", "--log_path", type=str, default="output/log_path")
    args = parser.parse_args()
    return args


def main(options):
    '''
    Set random seed
    '''
    if torch.cuda.is_available():
        torch.cuda.manual_seed(769)
    else:
        torch.manual_seed(769)
    '''
    Assuming that you are using the data in the GitHub repo.
    '''
    options.input = 'grouped_data/'

    '''
    Record experiment parameter settings.
    '''
    if not os.path.exists(options.output):
        os.makedirs(options.output)
    output_file = open(options.output + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(options)))
    
    training_params = {"batch_size": options.batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": options.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    '''
    Load the training set and validation set
    '''
    training_set = MyDataset(options.input + "train" + "_emotion_group.tsv", options.max_length)
    test_set = MyDataset(options.input + "dev" + "_emotion_group.tsv", options.max_length)
    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)

    '''
    Decide which model architecture to use
    '''
    if options.feature == "small":
        model = CharacterLevelCNN(input_length=options.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(options.alphabet),
                                  n_conv_filters=256, n_fc_neurons=1024)

    elif options.feature == "large":
        model = CharacterLevelCNN(input_length=options.max_length, n_classes=training_set.num_classes,
                                  input_dim=len(options.alphabet),
                                  n_conv_filters=1024, n_fc_neurons=2048)
    else:
        sys.exit("Invalid feature mode!")
    
    '''
    Tensorboard log
    '''
    options.dataset = 'GoEmotions'
    log_path = "{}_{}_{}".format(options.log_path, options.feature, options.dataset)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    '''
    Use cuda if supported
    '''
    if torch.cuda.is_available():
        model.cuda()
    '''
    Define loss function and optimizers
    '''
    criterion = nn.CrossEntropyLoss()
    if options.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=options.lr)
    elif options.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    best_loss = 1e5
    best_epoch = 0

    '''
    Start the training process
    '''
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(options.num_epochs):
        '''
        Iterate through the batches
        '''
        for iter, batch in enumerate(training_generator):
            feature, label = batch
            if torch.cuda.is_available():
                '''
                use cuda (transfer to GPU) if possible
                '''
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            # Forward pass
            predictions = model(feature)
            loss = criterion(predictions, label)
            # Backprop
            loss.backward()
            optimizer.step()
            # evaluate using CPU
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                options.num_epochs,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            # Record training behavior
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        
        '''
        Evaluate on the validation set.
        '''
        # evaluation mode
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for batch in test_generator:
            te_feature, te_label = batch
            num_sample = len(te_label)
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            with torch.no_grad():
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())

        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            options.num_epochs,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()
        if te_loss + options.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, "{}/char-cnn_{}_{}".format(options.output, options.dataset, options.feature))
        # Early stopping
        if epoch - best_epoch > options.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, te_loss, best_epoch))
            break
        # Learning rate schedule, /2 every i epochs
        if options.optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
    


if __name__ == "__main__":
    options = get_args()
    main(options)

