# BERTweet_Model

## Config setting
You can change any config in `configs.py` or directly add argument after `python main.py`.

The default mode used `vinai/bertweet-base` pretrained model.
If you would like to change it into BERT, just run the following command to train,

``python main.py --pretrained_model='bert-base-cased'``

## Test results
The `*.json` contains the test result we implemented on validation dataset and test dataset.