from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
import numpy as np

import VGGNet_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/media/oran/Data/ImageNet/train/',
                           """Path to the ImageNet data directory.""")
tf.app.flags.DEFINE_string('weights_dir', '/home/oran/python/VGGNet/vgg16_weights/',
                           """Path to vgg16 pre trained weights.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = VGGNet_input.IMAGE_SIZE
NUM_CLASSES = VGGNet_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = VGGNet_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = VGGNet_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 15.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0       # Initial learning rate.
# LR SHOULD BE SMALLER IF USING PRE TRAINED WEIGHTS!!!

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, layer_name):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    if FLAGS.use_pre_trained:
        if name=='weights':
            filename = FLAGS.weights_dir + layer_name + '_W.npy'
        else: #biases
            filename = FLAGS.weights_dir + layer_name + '_b.npy'
        init=tf.constant(np.load(filename), dtype=tf.float32)
        var = tf.get_variable(name, initializer=init, dtype=dtype)
    else:
        init=initializer
        var = tf.get_variable(name, shape, initializer=init, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, wd, layer_name):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
#  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.contrib.layers.xavier_initializer(uniform=False),
      layer_name)
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for ImageNet training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  images,labels = VGGNet_input.distorted_inputs(data_dir=FLAGS.data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for ImageNet evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = FLAGS.data_dir
  images, labels = VGGNet_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images, train=False):

  # conv1_1
  with tf.variable_scope('conv1_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 3, 64], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), layer_name=scope.name)
    conv1_1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv1_1)
  
  # conv1_2
  with tf.variable_scope('conv1_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0), layer_name=scope.name)
    conv1_2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv1_2)

  # pool1
  pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

  # conv2_1
  with tf.variable_scope('conv2_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), layer_name=scope.name)
    conv2_1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv2_1)
    
  # conv2_2
  with tf.variable_scope('conv2_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.0), layer_name=scope.name)
    conv2_2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv2_2)
    
  # pool2
  pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
  
  # conv3_1
  with tf.variable_scope('conv3_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), layer_name=scope.name)
    conv3_1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv3_1)
    
  # conv3_2
  with tf.variable_scope('conv3_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), layer_name=scope.name)
    conv3_2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv3_2)
      
  # conv3_3
  with tf.variable_scope('conv3_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.0), layer_name=scope.name)
    conv3_3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv3_3)
  
  # pool3
  pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
  
  # conv4_1
  with tf.variable_scope('conv4_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), layer_name=scope.name)
    conv4_1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv4_1)
    
  # conv4_2
  with tf.variable_scope('conv4_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), layer_name=scope.name)
    conv4_2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv4_2)
      
  # conv4_3
  with tf.variable_scope('conv4_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), layer_name=scope.name)
    conv4_3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv4_3)
  
  # pool4
  pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    
  # conv5_1
  with tf.variable_scope('conv5_1') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), layer_name=scope.name)
    conv5_1 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv5_1)
    
  # conv5_2
  with tf.variable_scope('conv5_2') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), layer_name=scope.name)
    conv5_2 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv5_2)
      
  # conv5_3
  with tf.variable_scope('conv5_3') as scope:
    kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], wd=0.000, layer_name=scope.name)
    conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.0), layer_name=scope.name)
    conv5_3 = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
    _activation_summary(conv5_3)
  
  # pool5
  pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

  # fc6
  with tf.variable_scope('fc6') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 4096], wd=0.000, layer_name=scope.name)
    biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0), layer_name=scope.name)
    fc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    if train:
        fc6 = tf.nn.dropout(fc6, 0.5)
    _activation_summary(fc6)

  # fc7
  with tf.variable_scope('fc7') as scope:
    weights = _variable_with_weight_decay('weights', shape=[4096, 4096], wd=0.000, layer_name=scope.name)
    biases = _variable_on_cpu('biases', [4096], tf.constant_initializer(0.0), layer_name=scope.name)
    fc7 = tf.nn.relu(tf.matmul(fc6, weights) + biases, name=scope.name)
    if train:
        fc7 = tf.nn.dropout(fc7, 0.5)
    _activation_summary(fc7)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [4096, NUM_CLASSES], wd=0.000, layer_name='fc8')
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0), layer_name='fc8')
    softmax_linear = tf.add(tf.matmul(fc7, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name + ' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.MomentumOptimizer(lr, 0.9) # GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
