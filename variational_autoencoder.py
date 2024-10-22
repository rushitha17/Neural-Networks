""" Variational Auto-Encoder Example.

Using a variational auto-encoder to generate digits images from noise.
MNIST handwritten digits are used as training examples.

References:
    - Auto-Encoding Variational Bayes The International Conference on Learning
    Representations (ICLR), Banff, 2014. D.P. Kingma, M. Welling
    - Understanding the difficulty of training deep feedforward neural networks.
    X Glorot, Y Bengio. Aistats 9, 249-256
    - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    - [VAE Paper] https://arxiv.org/abs/1312.6114
    - [Xavier Glorot Init](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf).
    - [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from keras import datasets

# Import MNIST data
mnist = datasets.mnist

# Load dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(-1, 784)

# Parameters
learning_rate = 0.001
num_steps = 30000
batch_size = 64

# Network Parameters
image_dim = 784  # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# A custom initialization (Xavier Glorot initialization)
def glorot_init(shape):
    return tf.random.normal(shape=shape, stddev=1. / tf.math.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Define the forward pass (encoder -> latent sampling -> decoder)
def encoder(input_image):
    layer_1 = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']
    layer_1 = tf.nn.tanh(layer_1)
    z_mean = tf.matmul(layer_1, weights['z_mean']) + biases['z_mean']
    z_std = tf.matmul(layer_1, weights['z_std']) + biases['z_std']
    return z_mean, z_std

def sample_latent(z_mean, z_std):
    eps = tf.random.normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
    return z_mean + tf.exp(z_std / 2) * eps

def decoder(z):
    layer_1 = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']
    layer_1 = tf.nn.tanh(layer_1)
    output = tf.matmul(layer_1, weights['decoder_out']) + biases['decoder_out']
    return tf.nn.sigmoid(output)

# Define VAE Loss
def vae_loss(x_reconstructed, x_true, z_mean, z_std):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.math.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.math.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

# Define optimizer
optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate)

# Training step
def train_step(batch_x):
    with tf.GradientTape() as tape:
        z_mean, z_std = encoder(batch_x)
        z = sample_latent(z_mean, z_std)
        reconstruction = decoder(z)
        loss = vae_loss(reconstruction, batch_x, z_mean, z_std)
    gradients = tape.gradient(loss, list(weights.values()) + list(biases.values()))
    optimizer.apply_gradients(zip(gradients, list(weights.values()) + list(biases.values())))
    return loss

# Start training
for i in range(1, num_steps + 1):
    # Get the next batch of MNIST data (only images are needed, not labels)
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    batch_x = x_train[idx]

    # Train step
    l = train_step(batch_x)
    if i % 1000 == 0 or i == 1:
        print(f'Step {i}, Loss: {l.numpy()}')

# Testing: Generate images from random noise
n = 20
x_axis = np.linspace(-3, 3, n)
y_axis = np.linspace(-3, 3, n)

canvas = np.empty((28 * n, 28 * n))
for i, yi in enumerate(x_axis):
    for j, xi in enumerate(y_axis):
        z_mu = np.array([[xi, yi]] * batch_size)
        #z_std to ensure it is float32
        z = sample_latent(z_mu, np.zeros_like(z_mu, dtype=np.float32))  # Assume std is 0 for testing
        x_mean = decoder(z).numpy()
        canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()