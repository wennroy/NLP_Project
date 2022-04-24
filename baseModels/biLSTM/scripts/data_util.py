import tensorflow as tf
import numpy as np

# the 28 emotions in our dataset
LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

# Construct training dataset & validation dataset needed for the biLSTM model
def process_label(features):
    """
    Preprocess function. This will be an entry for the map function of tf.data.dataset
    We create the label vector first.
    input:
      features, each entry in the dataset
    output:
      A dictionary (like the original input)
    """
    text = features["comment_text"]  # Text's key in GoEmotions
    label = tf.stack([features[label] for label in LABELS], axis=-1)
    label = tf.cast(label, tf.float32)
    model_features = (text, label)
    return model_features


def load_embeddings(embedding_dir):
    embedding_file = open(embedding_dir)
    embedding_dict = {}
    for row in embedding_file:
        values = row.split()
        word = values[0]
        embedding_vec = np.asarray(values[1:], dtype="float32")
        embedding_dict[word] = embedding_vec
    embedding_file.close()
    return embedding_dict
