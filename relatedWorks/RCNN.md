# Recurrent Convolutional Neural Networks for Text Classification

[link](https://ojs.aaai.org/index.php/AAAI/article/view/9513)

## Summary

The authors of this work design recurrent convolutional neural networks to extract contextual information. They suggest that their method introduces less noise than window-based neural networks. They conduct experiments on four datasets and report state-of-the-art results.

## Contributions

In their model architecture, the first part has a bi-directional recurrent structure. For an input word, their model represents it by a concatenation of the left context vector, word embedding, and right context vector. Then, the model projects the word representation to a latent semantic vector by a linear transformation and tanh activation function. 

The computed latent semantic vector is fed to the second part, a max-pooling layer. In some sense, the layer is finding the most representative semantic vector. The output layer takes the pooling result and applies the softmax function to give predictions.

