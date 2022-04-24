"""
This file serve as a script backup for the notebook
"""
# Load libraries
from json import load
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from data_util import LABELS, process_label, load_embeddings

"""
Load dataset from tfds
"""
train_ds = tfds.load("goemotions", split="train")
val_ds = tfds.load("goemotions", split="validation")
test_ds = tfds.load("goemotions", split="test")

# Create Vocabulary
# We use at most 50000 words
MAX_FEATURES = 50000
# Restrict the length of sentences to 256
MAX_LENGTH = 300

# We use keras' text vectorization layer
# like the Vocab class we used in assignment 2
vectorized_layer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES, output_sequence_length=MAX_LENGTH
)
# Add all the training vocabularies
vectorized_layer.adapt(train_ds.map(lambda text: text["comment_text"]))

# Create the training set, validation set
BUFFER_SIZE = 60000
BATCH_SIZE = 64

trial_train_ds = train_ds.map(
    process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False
)
trial_val_ds = val_ds.map(
    process_label, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False
)
train_dataset = (
    trial_train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)
val_dataset = trial_val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load embedding:
embedding_dir = "glove.twitter.27B.200d.txt"
EMBEDDING_DIM = 200
embedding_dict = load_embeddings(embedding_dir)
# Coordinate with the previous vocabulary
current_words = vectorized_layer.get_vocabulary()
embedding_weights = np.zeros((len(current_words), EMBEDDING_DIM))
for i, words in enumerate(current_words):
    embedding_word = embedding_dict.get(words)
    if embedding_word is not None:
        embedding_weights[i] = embedding_word

NUM_OF_CLASSES = 28
# Define the model
biLSTM_model = tf.keras.Sequential(
    [
        vectorized_layer,
        tf.keras.layers.Embedding(
            input_dim=len(vectorized_layer.get_vocabulary()),
            output_dim=200,
            mask_zero=True,
            weights=[embedding_weights],
            trainable=True,
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(NUM_OF_CLASSES),
    ]
)

# Learning rate and optimizer
lr = 1e-3
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

biLSTM_model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

# Use a callback to store best models.
checkpoint_dir = "/output"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

# Fit the model
history = biLSTM_model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    validation_steps=30,
    callbacks=[model_checkpoint_callback],
)
