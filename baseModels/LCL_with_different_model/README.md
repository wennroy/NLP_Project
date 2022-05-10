# LCL for different model setting

## Config Setting
You can change any config in `configs.py` or directly add argument after `python main.py`.

The default mode is used CharBERT model for Contextual model and ELECTRA model for Weighting model.

You can change the setting by running
 
 ```python main.py --pretrained_model='bert-base-cased' --weighting_model='imvladikon/charbert-bert-wiki'```(BERT+CharBERT)
 
 to change the default model. We support most of the BERT-related model, such as
 ```--pretrained_model='google/electra-base-discriminator'```(ELECTRA) etc.
 
 
## tSNE visualization
Make sure you have already save the model (a `.pt` file), then change the save_name into corresponding name for `.pt` file.
Then run ``python tSNE_visualization.py --argument``

`--argument` is the argument you set previously to run `main.py`.