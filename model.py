import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

#####################
# Unpack MNIST data #
#####################

fdir = Path("dataset/train_dataset/")

with open(fdir / "train-images.idx3-ubyte", 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrow, ncol = struct.unpack(">II", f.read(8))
    img_train = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    img_train = img_train.reshape((size, nrow, ncol))

with open(fdir / "train-labels.idx1-ubyte", 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    label_train = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    label_train = label_train.reshape((size, 1))

#########    
# Model #
#########
train_dataset = tf.data.Dataset.from_tensor_slices((img_train, label_train))
test_dataset = tf.data.Dataset.from_tensor_slices((img_train, label_train))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(train_dataset, epochs=10)

model.evaluate(test_dataset)
