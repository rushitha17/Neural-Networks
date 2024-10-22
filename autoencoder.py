""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, losses, models


# Load MNIST data using Keras
mnist = tf.keras.datasets.mnist

# Split into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to the range 0-1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((x_train.shape[0], -1))  # (60000, 784)
x_test = x_test.reshape((x_test.shape[0], -1))  # (10000, 784)


# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

'''display_step = 1000
examples_to_show = 10'''

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
#X = tf.placeholder("float", [None, num_input])
# Directly use NumPy arrays or tensors
X = tf.keras.Input(shape=(num_input,))

def build_autoencoder():
    input_img = layers.Input(shape=(num_input,))
    # Encoder
    x = layers.Dense(num_hidden_1, activation='sigmoid')(input_img)
    encoded = layers.Dense(num_hidden_2, activation='sigmoid')(x)
    
    # Decoder
    x = layers.Dense(num_hidden_1, activation='sigmoid')(encoded)
    decoded = layers.Dense(num_input, activation='sigmoid')(x)
    
    # Autoencoder model
    autoencoder = models.Model(input_img, decoded)
    return autoencoder

# Build the autoencoder
autoencoder = build_autoencoder()

# Compile the model
autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate), loss='mean_squared_error')

# Training the autoencoder
autoencoder.fit(x_train, x_train, 
                epochs=50, 
                batch_size=batch_size, 
                shuffle=True, 
                validation_data=(x_test, x_test))

# Testing and Visualization
n = 10  # Number of images to display
canvas_orig = np.empty((28, 28 * n))
canvas_recon = np.empty((28, 28 * n)) 

# Get a batch of test images
batch_x = x_test[:n]
# Encode and decode the digit image
g = autoencoder.predict(batch_x)

# Display original images
for i in range(n):
    canvas_orig[:, i * 28:(i + 1) * 28] = batch_x[i].reshape([28, 28])
# Display reconstructed images
for i in range(n):
    canvas_recon[:, i * 28:(i + 1) * 28] = g[i].reshape([28, 28])

# Plot the original images
plt.figure(figsize=(10, 2))
plt.title("Original Images")
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.axis('off')
plt.show()

# Plot the reconstructed images
plt.figure(figsize=(10, 2))
plt.title("Reconstructed Images")
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.axis('off')
plt.show()

'''weights = {
    'encoder_h1': tf.Variable(tf.random.normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random.normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random.normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random.normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random.normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random.normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random.normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random.normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    #layer_1 = layers.Dense(num_hidden_1, activation='sigmoid')(x)
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    #layer_2 = layers.Dense(num_hidden_2, activation='sigmoid')(layer_1)
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = layers.Dense(num_hidden_1, activation='sigmoid')(x)
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = layers.Dense(num_input, activation='sigmoid')(layer_1)
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# Use the MeanSquaredError loss function provided by Keras
mse_loss = losses.MeanSquaredError()

# In your training step, use the loss function directly
loss = mse_loss(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()'''
