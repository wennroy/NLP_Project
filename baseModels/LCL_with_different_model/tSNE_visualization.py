import torch
import torch.nn as nn
import numpy as np
from read_data import create_data, Goemotions
from torch.utils.data import Dataset, DataLoader
from model import contextual_encoder, weighting_network

import datetime
from types import SimpleNamespace
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.cm
import matplotlib.colors


def tsne_visualization(args):
    emotions_mapping = []
    with open(args.emotions_txt, "r", encoding="UTF-8") as fm:
        for line in fm:
            emotions_mapping.append(line.strip("\n"))
    print(f'Read from {args.emotions_txt}.')
    print(f'Mapping 0 to {len(emotions_mapping) - 1} to label {emotions_mapping}')

    print("Visualizing...")
    targets = []
    embedding_result = []
    single_targets = []
    single_embedding_result = []
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

        model.eval()
        test_data = create_data(args.test, 'dev')
        test_dataset = Goemotions(test_data, 28, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=test_dataset.collate_fn)

        for test_idx, (
                test_token, test_attn_mask, test_weighting_token, test_weighting_attn_mask,
                test_label) in enumerate(test_dataloader):

            test_token, test_attn_mask = test_token.to(device), test_attn_mask.to(device)
            normalized_embedding = model.model(test_token, test_attn_mask).last_hidden_state[:,0,:]
            embedding_result.extend(normalized_embedding.detach().cpu().numpy())
            labels = test_label.cpu().numpy()
            for i, label in enumerate(labels):
                if np.sum(label) == 1:
                    single_targets.append(np.argmax(label))
                    single_embedding_result.append(normalized_embedding[i,:].detach().cpu().numpy())
            targets.extend(labels)

        targets_with_labels = []
        single_targets_with_labels = []
        for target in targets:
            tmp_target = ''
            for index in range(28):
                if target[index] == 1:
                    tmp_target += str(emotions_mapping[index]) + "+"
            targets_with_labels.append(tmp_target[:-1])
        for target in single_targets:
            single_targets_with_labels.append(emotions_mapping[target])
    
    print(len(targets_with_labels))
    print(f'Number of single label samples is {len(single_targets_with_labels)}')
    print("Finished obtaining embedding features")
    embedding_result = np.array(embedding_result)
    n_samples, n_features = embedding_result.shape

    X_embedded = TSNE(n_components=2, verbose=1, perplexity=40, random_state=769).fit_transform(single_embedding_result)
    data = pd.DataFrame()
    print(X_embedded[:3,0])
    print(X_embedded[:3,1])
    data['X'] = X_embedded[:,0]
    data['Y'] = X_embedded[:,1]
    n_labels = len(np.unique(single_targets_with_labels))
    data['labels'] = single_targets_with_labels
    print(n_labels)
    print(data)
    plt.figure(figsize=(16, 10))
    mapping_orders = ["admiration","amusement","approval","excitement","caring","desire","gratitude","joy","love","optimism","pride","relief","realization","curiosity","surprise","neutral","confusion","nervousness","embarrassment","sadness","remorse","grief","disapproval","annoyance","fear","anger","disappointment","disgust"]
    mapping_orders.reverse()
    print(mapping_orders)
    sns.scatterplot(
        x='X', y='Y',
        hue="labels",
        palette=sns.color_palette("Spectral", n_colors=n_labels),
        hue_order=mapping_orders,
        data=data,
        legend="auto",
        alpha=0.6
    )
    plt.savefig("TSNE_plot.pdf")

if __name__ == "__main__":
    from configs import get_args
    save_name = '16-4e-05-9-tsv-charbert-bert-wiki.pt'
    save_name = '16-4e-05-9-tsv-bert-base-cased.pt'
    args = get_args()
    args.filepath = save_name
    tsne_visualization(args)


