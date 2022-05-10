# BERT and BERT-related model Direct Implementation

## Config setting
You can change any config in `configs.py` or directly add argument after `python main.py`.

The default mode used `vinai/bertweet-base`(BERTweet) pretrained model.
If you would like to change it into any BERT related models you like, just run the following command to train,

``python main.py --pretrained_model='bert-base-cased'``(BERT)

We also run `--pretrained_model='imvladikon/charbert-bert-wiki'`(CharBERT) or `--pretrained_model='google/electra-base-discriminator'`(ELECTRA) as our pretrained_model.

## Results
The `*.json` contains the test result we implemented on validation dataset and test dataset.