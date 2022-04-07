# Character-level Convolutional Networks for Text Classification

[link](https://arxiv.org/pdf/1509.01626.pdf)

## Summary
In this work, the authors examine the use of character-level convolutional neural networks for text classification. The authors treat text as a kind of raw signal at the character level and try to extract information by 1-dimensional convolutions. They evaluate their model on large scale datasets and obtain competitve results.

## Contributions
The key component they have built is a 1-D convolution module. They define a discrete input function g and a kernel function f. The convolution is conducted on g and f. Another important component is the temporal or 1-D max-pooling layer. Their empirical results show that this pooling layer is essential for training the model.

The authors propose a way to encode characters in a text. They only consider a finite set of characters with length m. Then, each character is represented in an 'one-hot' encoding manner. In the original paper, English alphabets, 10 digits, and 33 other characters are included. Characters that are not in the set would be assigned to vectors with all zeros. Also, they only consider text sequence that has a length smaller than a determined upper bound. 

We will provide more details in the following section, as our experiments involve this character-level CNN model. 

