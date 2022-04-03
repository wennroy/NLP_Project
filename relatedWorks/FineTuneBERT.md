# How to Fine-Tune BERT for Text Classification?

[Link](https://arxiv.org/pdf/1905.05583.pdf)

## Summary

Bidirectional Encoder Representations from Transformers (BERT) is a novel and powerful language representation model. Researchers observe great performance in multiple language analysis tasks by fine-tuning a pre-trained BERT model. In this paper, the authors investigate different fine-tuning methods and offer their suggestions.

## Contributions

The authors propose a three-step solution for finetuning the pre-trained BERT model. The first step is further pre-training on the target dataset or related-domain dataset. The second step is optional and involves multitask learning approaches. The final step is fine-tuning on the target task.

The authors conduct thorough experiments on eight text classification datasets. They suggest that the top layer of BERT is more helpful for text classification. Also, they show that the first step of their three-step solution significantly improves the model's performance.