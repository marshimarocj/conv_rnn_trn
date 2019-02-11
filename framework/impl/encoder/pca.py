import enum

import tensorflow as tf

import framework.model.module


class Config(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.dim_ft = 1024
    self.dim_output = 512 # dim of feature layer output

  def _assert(self):
    pass


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'vanilla.Encoder'

  class InKey(enum.Enum):
    FT = 'ft' # (None, dim_ft)

  class OutKey(enum.Enum):
    EMBED = 'ft_embed' # (None, dim_output)

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_input = self._config.dim_ft
      self.fc_W = tf.contrib.framework.model_variable('fc_W',
        shape=(dim_input, self._config.dim_output), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self._weights.append(self.fc_W)
      self.fc_B = tf.contrib.framework.model_variable('fc_B',
        shape=(self._config.dim_output,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.fc_B)

  def _embed(self, in_ops):
    ft = in_ops[self.InKey.FT]

    embed = tf.nn.xw_plus_b(ft, self.fc_W, self.fc_B)

    return embed

  def get_out_ops_in_mode(self, in_ops, mode, reuse=True, **kwargs):
    with tf.variable_scope(self.name_scope):
      embed_op = self._embed(in_ops)
    return {
      self.OutKey.EMBED: embed_op,
    }


class Encoder1D(framework.model.module.AbstractModule):
  name_scope = 'pca.Encoder1D'

  class InKey(enum.Enum):
    FT = 'ft' # (None, dim_ft)

  class OutKey(enum.Enum):
    EMBED = 'ft_embed' # (None, dim_output)

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      dim_input = self._config.dim_ft
      self.fc_W = tf.contrib.framework.model_variable('fc_W',
        shape=(1, dim_input, self._config.dim_output), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self._weights.append(self.fc_W)
      self.fc_B = tf.contrib.framework.model_variable('fc_B',
        shape=(self._config.dim_output,), dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.1, 0.1))
      self._weights.append(self.fc_B)

  def _embed(self, in_ops):
    ft = in_ops[self.InKey.FT]

    embed = tf.nn.conv1d(ft, self.fc_W, 1, 'VALID')
    embed = tf.nn.bias_add(embed, self.fc_B)

    return embed

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      embed = self._embed(in_ops)
    return {
      self.OutKey.EMBED: embed,
    }


class Encoder2D(framework.model.module.AbstractModule):
  name_scope = 'pca.Encoder2D'

  class InKey(enum.Enum):
    FT = 'ft' # (None, dim_time, H, W, dim_ft)

  class OutKey(enum.Enum):
    EMBED = 'embed' # (None, dim_time, H, W, dim_output)

  def _set_submods(self):
    return {}

  def _build_parameter_graph(self):
    with tf.variable_scope(self.name_scope):
      self.conv_W = tf.contrib.framework.model_variable('conv_W',
        shape=(1, 1, self._config.dim_ft, self._config.dim_output), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self._weights.append(self.conv_W)
      self.conv_B = tf.contrib.framework.model_variable('conv_B',
        shape=(self._config.dim_output,), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
      self._weights.append(self.conv_B)

  def _embed(self, ft):
    shape = tf.shape(ft)
    ft = tf.reshape(ft, [-1, shape[2], shape[3], shape[4]]) # (None*dim_time, H, W, dim_ft)
    ft = tf.nn.conv2d(ft, self.conv_W, [1, 1, 1, 1], 'VALID')
    ft = tf.nn.bias_add(ft, self.conv_B)
    ft = tf.reshape(ft, [-1, shape[1], shape[2], shape[3], self._config.dim_output])
    return ft

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      embed = self._embed(in_ops[self.InKey.FT])
      return {
        self.OutKey.EMBED: embed,
      }
