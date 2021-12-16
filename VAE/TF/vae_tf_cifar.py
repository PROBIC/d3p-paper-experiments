# Started from an example in
# https://danijar.com/building-variational-auto-encoders-in-tensorflow/

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_probability as tfp
tfd = tfp.distributions

import tensorflow as tf2

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

from tensorflow_privacy.privacy.optimizers import dp_optimizer_vectorized

import tensorflow_datasets as tfds

import argparse

parser = argparse.ArgumentParser(description="parse args")
parser.add_argument('-lr', '--learning-rate', default=1.0e-4, type=float, help='learning rate')
parser.add_argument('--dp_scale', default=1.5, type=float, help='scale for DP perturbations')
parser.add_argument('--no_dp', default=False, action='store_true', help='Use plain SVI instead of DPSVI algorithm')
parser.add_argument('--seed', default=0, type=int, help='seed')
args = parser.parse_args()

np.random.seed(args.seed)
tf2.random.set_seed(args.seed)
tf.set_random_seed(args.seed)

kernel_num = 128
channel_num = 3
image_size = 32
feature_size = image_size // 8


def make_encoder_dense(data, code_size, hidden_dim=400):
  x = tf.layers.flatten(data)
  x = tf.layers.dense(x, hidden_dim, tf.nn.softplus)
  loc = tf.layers.dense(x, code_size)
  scale = tf.layers.dense(x, code_size, tf.exp)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_encoder(data, code_size, hidden_dim=400):
  x = data
  x = tf.layers.conv2d(x, kernel_num // 4, kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.softplus)
  x = tf.layers.conv2d(x, kernel_num // 2, kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.softplus)
  x = tf.layers.conv2d(x, kernel_num     , kernel_size=(4,4), strides=(2,2), padding='same', activation=tf.nn.softplus)
  x = tf.layers.flatten(x)
  loc = tf.layers.dense(x, code_size)
  scale = tf.layers.dense(x, code_size, tf.exp)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_prior(code_size):
  loc = tf.zeros(code_size)
  scale = tf.ones(code_size)
  return tfd.MultivariateNormalDiag(loc, scale)


def make_decoder_dense(code, data_shape, hidden_dim=400):
  x = code
  x = tf.layers.dense(x, hidden_dim, tf.nn.softplus)
  logit = tf.layers.dense(x, np.prod(data_shape))
  logit = tf.reshape(logit, [-1] + data_shape)
  return tfd.Independent(tfd.Bernoulli(logit), 2)


def make_decoder(code, data_shape, hidden_dim=400):
  x = code
  x = tf.layers.dense(x, kernel_num * feature_size**2)
  x = tf.reshape(x, [-1, feature_size, feature_size, kernel_num])
  x = tf.layers.conv2d_transpose(x, kernel_num // 2, kernel_size = (4,4), strides=(2,2), padding='same', activation=tf.nn.softplus)
  x = tf.layers.conv2d_transpose(x, kernel_num // 4, kernel_size = (4,4), strides=(2,2), padding='same', activation=tf.nn.softplus)
  x = tf.layers.conv2d_transpose(x, channel_num,     kernel_size = (4,4), strides=(2,2), padding='same', activation=tf.nn.sigmoid)
  x = tfd.Independent(tfd.Bernoulli(x), 3)
  return x

def plot_codes(ax, codes, labels):
  ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
  ax.set_aspect('equal')
  ax.set_xlim(codes.min() - .1, codes.max() + .1)
  ax.set_ylim(codes.min() - .1, codes.max() + .1)
  ax.tick_params(
      axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')


def plot_samples(ax, samples):
  for index, sample in enumerate(samples):
    ax[index].imshow(sample, cmap='gray')
    ax[index].axis('off')


def normalize_image(ex):
  ex['image'] = tf.to_float(ex['image']) / 255.
  return ex


# Loaders
batch_size = 128

ds, auxinfo = tfds.load('cifar10', split='train', with_info=True)
num_examples = auxinfo.splits['train'].num_examples
ds = ds.shuffle(seed=args.seed, buffer_size=60000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
ds = ds.map(normalize_image).repeat()
iterator = tf.data.make_initializable_iterator(ds)
data = iterator.get_next()['image']

dstest = tfds.load('mnist', split='test')
dstest = dstest.shuffle(seed=args.seed, buffer_size=10000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
dstest = dstest.map(normalize_image)

testiterator = tf.data.make_initializable_iterator(ds)
testdata = iterator.get_next()['image']

make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

def model(data):
  # Define the model.
  prior = make_prior(code_size=50)
  posterior = make_encoder(data, code_size=50)
  code = posterior.sample()

  # Define the loss.
  likelihood = make_decoder(code, [32, 32, 3]).log_prob(data)
  divergence = tfd.kl_divergence(posterior, prior)
  elbo = likelihood - divergence
  samples = make_decoder(prior.sample(1), [32, 32, 3]).mean()
  return elbo, code, samples

elbo, code, samples = model(data)
telbo, tcode, tsamples = model(testdata)

if args.no_dp:
    optimize = tf.train.AdamOptimizer(args.learning_rate).minimize(-elbo)
else:
    clipping_threshold = 10.0

    optimizer = dp_optimizer_vectorized.VectorizedDPAdam(
    l2_norm_clip=clipping_threshold,
    noise_multiplier=args.dp_scale,
    num_microbatches=None, # do not aggregate samples in microbatches -> per-example gradients
    learning_rate=args.learning_rate)

    optimize = optimizer.minimize(-elbo)

fig, ax = plt.subplots(nrows=20, ncols=11, figsize=(10, 20))

tstart = time.time()
with tf.train.MonitoredSession() as sess:
  sess.run(iterator.initializer)
  sess.run(testiterator.initializer)
  for epoch in range(20):
    tstart = time.time()
    for i in range(num_examples//batch_size):
      sess.run(optimize)
    wt = time.time() - tstart
    aveloss = []
    for _ in range(10000//batch_size):
      test_elbo, test_codes, test_samples = sess.run([telbo, tcode, tsamples])
      aveloss.append(np.mean(test_elbo))
    print('Epoch', epoch, 'elbo', -np.mean(test_elbo), 'time', wt)
