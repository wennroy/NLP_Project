# BERTweet_Model

## Config setting
You can change any config in `configs.py` or directly add argument after `python main.py`.

The default mode used `vinai/bertweet-base` pretrained model.
If you would like to change it into any BERT related models you like, just run the following command to train,

``python main.py --pretrained_model='bert-base-cased'``

We also run `--pretrained_model='imvladikon/charbert-bert-wiki'` as our pretrained_model.

## Test results
The `*.json` contains the test result we implemented on validation dataset and test dataset.