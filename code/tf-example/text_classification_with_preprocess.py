# Movies comments text classification with preprocessing.

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
#print(tf.__version__)

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0]) # first comment

# map word to index
word_index = imdb.get_word_index()
len(train_data[0])
len(train_data[1])

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
decode_review(train_data[0]) # show the first comment

# regularization the lenght of comments
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

len(train_data[0])
len(train_data[1])
print(train_data[0])


# build model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

#model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#              loss='binary_crossentropy', # loss: 0.3386 - accuracy: 0.8712
#              loss='mean_squared_error', # loss: 0.0936 - accuracy: 0.8736
#              metrics=['accuracy'])

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='mean_squared_error', # loss: 0.0925 - accuracy: 0.8756
#              loss='binary_crossentropy', # loss: 0.3206 - accuracy: 0.8756
              metrics=['accuracy'])

# develop set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)


# create a plot graph
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

def make_graph():
    plt.plot(epochs, loss, 'k.', label='Training loss')
    plt.plot(epochs, val_loss, 'k', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.clf()
    plt.plot(epochs, acc, 'k.', label='Training acc')
    plt.plot(epochs, val_acc, 'k', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

make_graph()
