import tensorflow as tf
import numpy as np

w=tf.Variable(0, dtype = tf.float32)
cost = w ** 2 - w * 10 + 25
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

session.run(train)
print(session.run(w))

for i in range(1000):
    session.run(train)
print(session.run(w))
