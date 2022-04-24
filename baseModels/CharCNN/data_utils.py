"""
This file contains the helper methods we use for data preprocessing.
Some of the methods are not used in main.py. 
"""
import pandas as pd
import numpy as np
import sys
import csv
from torch.utils.data import Dataset

# Create datasets based on different grouping methods.
def generate_group_dataset(input_data, labels, json_file):
    """
    input:
      input_data: a pandas dataframe, e.g the dataset in our GitHub repo
      labels: the list of classes (in our case, emotions)
      json_file: encode the grouping method
    output:
      output_data: the grouped dataset.
      outout_labels: the label assignment
    """
    output_labels = {"neutral": 0}
    k = 1
    for group in json_file:
        output_labels[group] = k
        k += 1
    col_titles = [1, 0]
    output_data = input_data.reindex(columns=col_titles)
    for l in range(output_data.shape[0]):
        curLabel = labels[int(output_data[1][l].split(",")[0])]
        if curLabel == "neutral":
            output_data[1][l] = 0
        else:
            for group in json_file:
                if curLabel in json_file[group]:
                    output_data[1][l] = output_labels[group]
                    break
    return output_data, output_labels


class MyDataset(Dataset):
    """
    Use MyDataset class to generate torch dataset.
    """

    def __init__(self, data_path, max_length=1014):
        self.data_path = data_path
        # The authors only consider the following characters.
        self.vocabulary = list(
            """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""
        )
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []
        with open(data_path) as csv_file:
            reader = csv.reader(csv_file)
            for idx, line in enumerate(reader):
                text = ""
                for tx in line[1:]:
                    text += tx
                    text += " "
                """
                We assume that the data file has a specific format:
                  The first column: label, integer.
                  The second column: text data
                """
                label = int(line[0])
                texts.append(text)
                labels.append(label)
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        # Encode the characters as described in the paper.
        data = np.array(
            [
                self.identity_mat[self.vocabulary.index(i)]
                for i in list(raw_text)
                if i in self.vocabulary
            ],
            dtype=np.float32,
        )
        if len(data) > self.max_length:
            data = data[: self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (
                    data,
                    np.zeros(
                        (self.max_length - len(data), len(self.vocabulary)),
                        dtype=np.float32,
                    ),
                )
            )
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label
