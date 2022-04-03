# Deep Convolutional Neural Networks for Sentiment Analysis of Short Texts

[Link](https://aclanthology.org/C14-1008.pdf)

## Summary

Analyzing sentiments of short texts is challenging since there is less contextual information. In this work, the authors design a deep convolutional neural network to extract character- to sentence-level information. Their method achieves better performance than other models  on the Stanford Twitter Sentiment (STS) corpus and the Stanford Sentiment Treebank (SSTb).

## Contributions

They introduce an innovative network architecture (CharSCNN) that uses two convolutional layers to handle sentences of any size. Their network extracts feature from character-level to sentence-level.

CharSCNN transforms a word into a joint embedding of words and characters. After determining the embeddings for all the words in a sentence, CharSCNN extracts a sentence-level representation or 'global' feature vector. The network then makes predictions based on the 'global' feature vector.

In the binary classification tas, CharSCNN achieves better performance than the state-of-the-art model. However, its improvement on the fine-grained task is not that obvious.
