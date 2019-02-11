import enum
import math

import tensorflow as tf

import framework.model.module


class CellConfig(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_hidden = 128
    self.dim_input = 128
    self.keepout_prob = 1.
    self.keepin_prob = 1.
    self.conv_kernel = []

  def _assert(self):
    pass


class GRUCell(framework.model.module.AbstractModule):
  name_scope = 'conv_cell.GRUCell'

  class InKey(enum.Enum):
    INPUT = 'input' # (None, dim_input)
    STATE = 'state' # ((None, dim_hidden), (None, dim_hidden))
    IS_TRN = 'is_training'

  class OutKey(enum.Enum):
    OUTPUT = 'output'
    STATE = 'state'

  @property
  def state_size(self):
    return self._config.dim_hidden

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    kernel = self._config.conv_kernel
    with tf.variable_scope(self.name_scope):
      stddev = 1 / math.sqrt(self._config.dim_hidden + self._config.dim_input)
      self.gate_W = tf.contrib.framework.model_variable('gate_W', 
        shape=(kernel[0], kernel[1], self._config.dim_hidden + self._config.dim_input, 2*self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self.gate_b = tf.contrib.framework.model_variable('gate_b',
        shape=(2*self._config.dim_hidden,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0, dtype=tf.float32))
      self.candidate_W = tf.contrib.framework.model_variable('candidate_W',
        shape=(kernel[0], kernel[1], self._config.dim_hidden + self._config.dim_input, self._config.dim_hidden),
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
      self.candidate_b = tf.contrib.framework.model_variable('candidate_b',
        shape=(self._config.dim_hidden,),
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0, dtype=tf.float32))
      self._weights.append(self.gate_W)
      self._weights.append(self.gate_b)
      self._weights.append(self.candidate_W)
      self._weights.append(self.candidate_b)

  def _step(self, input, state, is_training):
    shape = tf.shape(input)
    input = tf.contrib.layers.dropout(input, 
      keep_prob=self._config.keepin_prob, noise_shape=[shape[0], 1, 1, shape[-1]], is_training=is_training)

    gate_inputs = tf.nn.conv2d(tf.concat([input, state], -1), self.gate_W, [1, 1, 1, 1], 'SAME') # (None, H, W, 4*dim_hidden)
    gate_inputs = tf.nn.bias_add(gate_inputs, self.gate_b)
    
    value = tf.sigmoid(gate_inputs)
    r, u = tf.split(value, num_or_size_splits=2, axis=-1)

    r_state = r * state

    candidate = tf.nn.conv2d(tf.concat([input, r_state], -1), self.candidate_W, [1, 1, 1, 1], 'SAME')
    candidate = tf.nn.bias_add(candidate, self.candidate_b)

    c = tf.tanh(candidate)
    new_h = u * state + (1 - u) * c
    return new_h

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      output = self._step(in_ops[self.InKey.INPUT], in_ops[self.InKey.STATE], in_ops[self.InKey.IS_TRN])
    return {
      self.OutKey.OUTPUT: output,
      self.OutKey.STATE: output,
    }
