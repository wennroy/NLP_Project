import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


def create_data(filename, flag='train'):
    data = []
    all_labels = []
    if filename[-3:] == 'csv':
        with open(filename, "r", encoding="UTF=8") as fp:
            csv_reader = csv.reader(fp)
            for line in csv_reader:
                text = line[0]
                emo_label_str = line[1]
                emo_label_str = emo_label_str.split(',')
                emo_label = []
                for i in emo_label_str:
                    all_labels.append(int(i))
                    emo_label.append(int(i))

                data.append((text, emo_label))
    elif filename[-3:] == 'tsv':
        with open(filename, "r", encoding="UTF-8") as fp:
            for line in fp:
                line = line.split("\t")
                emo_label_str = line[1]
                text = line[0]
                emo_label_str = emo_label_str.split(',')
                emo_label = []
                for i in emo_label_str:
                    all_labels.append(int(i))
                    emo_label.append(int(i))

                data.append((text, emo_label))
    num_labels = max(all_labels) + 1
    print(f"load {len(data)} data from {filename} for {flag} dataset, total {num_labels} labels")
    if flag == "train":
        return data, num_labels
    else:
        return data


class Goemotions(Dataset):
    def __init__(self, dataset, num_labels, args, sent_len=128):
        self.dataset = dataset
        self.p = args
        self.num_labels = num_labels
        self.sent_len = sent_len
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    def __len__(self):
        return len(self.dataset)

    def one_hot_label(self, data):
        label = data[1]
        label_vec = [0] * self.num_labels

        for i in label:
            label_vec[i] = 1.

        return label_vec

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        text = ele[0]
        label = self.one_hot_label(ele)
        return text, label

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        labels = torch.tensor(batch[1])
        texts = list(batch[0])
        tokens = torch.tensor(
            self.tokenizer(texts, padding=True, truncation=True, max_length=self.sent_len)["input_ids"])
        del batch
        return tokens, labels


if __name__ == '__main__':
    ## test
    from configs import get_args
    args = get_args()
    for filename in ["../../data/train.tsv", '../../data/goemotions_train.csv']:
        train_data, num_labels = create_data(filename, 'train')
        train_dataset = Goemotions(train_data, num_labels, args)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=train_dataset.collate_fn)
        for batch_idx, (batch_input, batch_label) in enumerate(train_dataloader):
            print(batch_idx)
            print(batch_input)
            print(batch_label)
            break