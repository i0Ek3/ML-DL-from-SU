# !!! YOU SHOULD KNOW !!!
# This is binay classify problem which use tf-hub and keras to solve it.
# And the data from Internet Movie Database and IMDB dataset, 50000 comments text in total, we use 25000 as train, 25000 as test.
# You need GPU else it failed.
# Ref: https://www.tensorflow.org/tutorials/keras/text_classification_with_hub?hl=zh-cn

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

#print("Version: ", tf.__version__)
#print("Eager mode: ", tf.executing_eagerly())
#print("Hub version: ", hub.__version__)
#print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
#(train_data, validation_data), test_data = tfds.load(name="imdb_reviews", split=(train_validation_split, tfds.Split.TEST), as_supervised=True)
(train_data, validation_data, test_data) = tfds.load(name="imdb_reviews", split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
train_labels_batch

# build and compile the model
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train the model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# evaluate the model
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
