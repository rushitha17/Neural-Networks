""" Neural Network with Eager API.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow's Eager API. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

import tensorflow as tf
from keras import datasets

mnist = datasets.mnist.load_data(path="mnist.npz")

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# Preprocess the data
(train_images, train_labels), (test_images, test_labels) = mnist
train_images = train_images.reshape(-1, num_input).astype('float32') / 255.0
test_images = test_images.reshape(-1, num_input).astype('float32') / 255.0

# Using TF Dataset to split data into batches
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(60000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# Define the neural network. To use eager API and tf.layers API together,
# we must instantiate a tfe.Network class as follow:
class NeuralNet(tf.keras.Model):
    def __init__(self):
        # Define each layer
        super(NeuralNet, self).__init__()
        # Hidden fully connected layer with 256 neurons
        self.layer1 = tf.keras.layers.Dense(n_hidden_1, activation='relu')
        # Hidden fully connected layer with 256 neurons
        self.layer2 = tf.keras.layers.Dense(n_hidden_2, activation='relu')
        # Output fully connected layer with a neuron for each class
        self.out_layer = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)


neural_net = NeuralNet()


# Cross-Entropy loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Calculate accuracy
accuracy_fn = tf.keras.metrics.SparseCategoricalAccuracy()


# SGD Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for step in range(num_steps):

    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            logits = neural_net(x_batch)
            batch_loss = loss_fn(y_batch, logits)

        # Compute gradients and update weights
        gradients = tape.gradient(batch_loss, neural_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))

        # Update the accuracy metric
        accuracy_fn.update_state(y_batch, logits)

    if (step + 1) % display_step == 0 or step == 0:
        print(f"Step: {step + 1}, Loss: {batch_loss.numpy():.9f}, Accuracy: {accuracy_fn.result().numpy():.4f}")

# Evaluate model on the test image set
test_acc = accuracy_fn(test_labels, neural_net(test_images))
print("Testset Accuracy: {:.4f}".format(test_acc))

'''# Iterate through the dataset
    d = dataset_iter.next()

    # Images
    x_batch = d[0]
    # Labels
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # Compute the batch loss
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss
    # Compute the batch accuracy
    batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        # Display the initial cost, before optimizing
        print("Initial loss= {:.9f}".format(average_loss))

    # Update the variables following gradients info
    optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))

    # Display info
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

# Evaluate model on the test image set
testX = mnist.test.images
testY = mnist.test.labels

test_acc = accuracy_fn(neural_net, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))
'''