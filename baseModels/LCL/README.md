# LCL for the same model setting

## Config Setting
You can change any config in `configs.py` or directly add argument after `python main.py`.

The default mode is used ELECTRA model for both Contextual model and Weighting model.

You can change the setting by running
 
 ```python main.py --pretrained_model='bert-base-cased'```(BERT)
 
 to change the default model. We support most of the BERT-related model, such as
 ```--pretrained_model='imvladikon/charbert-bert-wiki'```(CharBERT) etc.
 