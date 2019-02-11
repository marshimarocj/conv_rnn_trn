import json
import enum

import tensorflow as tf
import numpy as np


class Mode(enum.Enum):
  TRN_VAL = 0
  TST = 1
  ROLLOUT = 2
  SCORE = 3
  TRN = 4
  VAL = 5


class ModuleConfig(object):
  def __init__(self):
    self.subcfgs = {}
    self.freeze = False
    self.clip = False
    self.clip_interval = [-1., 1.]
    self.lr_mult = 1.0
    self.opt_alg = 'Adam'
    self.device = '/device:GPU:0'

  def load(self, cfg_dict):
    for key in cfg_dict:
      if key == 'subcfgs': # recursion
        data = cfg_dict[key]
        for key in data:
          self.subcfgs[key].load(data[key])
      elif key in self.__dict__:
        setattr(self, key, cfg_dict[key])

    self._assert()

  def save(self):
    out = {}
    for attr in self.__dict__:
      if attr == 'subcfgs':
        cfgs = self.__dict__[attr]
        out['subcfgs'] = {}
        for key in cfgs:
          out['subcfgs'][key] = cfgs[key].save()
      else:
        val = self.__dict__[attr]
        if type(val) is not np.ndarray: # ignore nparray fields, which are used to initialize weights
          out[attr] = self.__dict__[attr]
    return out

  def _assert(self):
    """
    check compatibility between configs
    """
    pass


class AbstractModule(object):
  name_scope = 'AbstractModule'

  class InKey(enum.Enum):
    pass

  class OutKey(enum.Enum):
    pass

  def __init__(self, config):
    self._config = config
    self._op2monitor = {}
    self._submods = self._set_submods()
    self._weights = []

  @property
  def config(self):
    return self._config

  @property
  def op2monitor(self):
    return self._op2monitor

  @property
  def submods(self):
    return self._submods

  @property
  def weights(self):
    return self._weights

  def _set_submods(self):
    """
    return a dictionary of submods
    """
    raise NotImplementedError("""please customize AbstractModule._set_submods""")

  def _build_parameter_graph(self):
    """
    this would be called before get_out_ops_in_mode. 
    shared parts between trn and tst are encouraged to be placed in this function
    """
    raise NotImplementedError("""please customize AbstractModule.build_parameter_graph""")

  def get_out_ops_in_mode(self, in_ops, mode, reuse=False, **kwargs):
    """
    return out_ops (a dictionary) given in_ops (a dictionary), 
    """
    raise NotImplementedError("""please customize AbstractModule.get_out_ops_in_mode""")

  def build_parameter_graph(self):
    with tf.device(self._config.device):
      self._build_parameter_graph()
    for key in self._submods:
      submod = self._submods[key]
      with tf.device(self._config.subcfgs[key].device):
        submod.build_parameter_graph()


class ModelConfig(ModuleConfig):
  def __init__(self):
    ModuleConfig.__init__(self)

    self.trn_batch_size = 256
    self.tst_batch_size = 128
    self.num_epoch = 100
    self.val_iter = 100
    self.val_loss = True
    self.monitor_iter = 1
    self.base_lr = 1e-4
    self.decay_boundarys = []
    self.decay_values = []

  def load(self, file):
    with open(file) as f:
      data = json.load(f)
      ModuleConfig.load(self, data)

  def save(self, out_file):
    out = ModuleConfig.save(self)
    with open(out_file, 'w') as fout:
      json.dump(out, fout, indent=2)


class AbstractModel(AbstractModule):
  name_scope = 'AbstractModel'

  class DefaultKey(enum.Enum):
    INIT = 'init'
    LOSS = 'loss'
    TRAIN = 'train'
    SAVER = 'saver'
    SUMMARY = 'summary'

  def __init__(self, config):
    AbstractModule.__init__(self, config)

    self._inputs = {}
    self._outputs = {}

  def _add_input_in_mode(self, mode):
    """
    return dictionary of inputs
    """
    raise NotImplementedError("""please customize AbstractModel._add_input_in_mode""")

  def _add_loss(self):
    """
    return loss op
    """
    raise NotImplementedError("""please customize AbstractModel._add_loss""")

  def op_in_tst(self, **kwargs):
    """
    return dictionary of op in tst
    """
    raise NotImplementedError("""please customize AbstractModel.op_in_tst""")

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def init_op(self):
    return self._outputs[self.DefaultKey.INIT]

  @property
  def saver(self):
    return self._outputs[self.DefaultKey.SAVER]

  @property
  def summary_op(self):
    return self._outputs[self.DefaultKey.SUMMARY]

  def build_trn_tst_graph(self, decay_boundarys=[], step=0):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TRN_VAL)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TRN_VAL)

      if len(decay_boundarys) > 0:
        global_step = tf.Variable(step, trainable=False)
        base_lr = tf.train.piecewise_constant(global_step, decay_boundarys, self._config.decay_values)
      else:
        base_lr = self._config.base_lr

      self._outputs[self.DefaultKey.LOSS] = self._add_loss()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        self._outputs[self.DefaultKey.TRAIN] = self._calculate_gradient(base_lr)

      _recursive_gather_op2monitor_helper(self, self._op2monitor)
      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.SUMMARY] = self._add_summary()
      self._outputs[self.DefaultKey.INIT] = self._add_init()

    return basegraph

  def build_tst_graph(self):
    basegraph = tf.Graph()
    with basegraph.as_default():
      self._inputs = self._add_input_in_mode(Mode.TST)
      self.build_parameter_graph()
      self._outputs = self.get_out_ops_in_mode(self._inputs, Mode.TST)

      self._outputs[self.DefaultKey.SAVER] = self._add_saver()
      self._outputs[self.DefaultKey.INIT] = self._add_init()

    return basegraph

  def op_in_trn(self, **kwargs):
    return {
      self.DefaultKey.LOSS: self._outputs[self.DefaultKey.LOSS],
      self.DefaultKey.TRAIN: self._outputs[self.DefaultKey.TRAIN],
    }

  def op_in_val(self, **kwargs):
    return {
      self.DefaultKey.LOSS: self._outputs[self.DefaultKey.LOSS],
    }

  def _add_summary(self):
    with tf.variable_scope(self.name_scope):
      tf.summary.scalar('loss', self._outputs[self.DefaultKey.LOSS])
      for var in tf.trainable_variables():
        tf.summary.histogram(var.name + '/activations', var)
      summary_op = tf.summary.merge_all()
    return summary_op

  def _add_saver(self):
    model_vars = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    saver = tf.train.Saver(model_vars, max_to_keep=1000)
    return saver

  def _add_init(self):
    with tf.variable_scope(self.name_scope):
      init = tf.global_variables_initializer()
    return init

  def _calculate_gradient(self, base_lr):
    loss_op = self._outputs[self.DefaultKey.LOSS]

    train_ops = _recursive_train_ops(self, base_lr, loss_op, 
      save_memory=self._config.save_memory)
    return train_ops


def _recursive_train_ops(module, base_lr, loss_op, save_memory=False):
  weights = module.weights

  all_train_ops = []
  if len(weights) > 0 and not module.config.freeze:
    for weight in weights:
      learning_rate = base_lr * module.config.lr_mult

      if module.config.opt_alg == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)

      grads = tf.gradients(loss_op, [weight], gate_gradients=True)
      print grads, weight
      train_op = optimizer.apply_gradients([(grads[0], weight)])
      all_train_ops.append(train_op)

  # recursive
  for key in module.submods:
    submod = module.submods[key]
    train_ops = _recursive_train_ops(submod, base_lr, loss_op, save_memory=save_memory)
    all_train_ops += train_ops

  return all_train_ops


def _recursive_gather_op2monitor_helper(module, op2monitor):
  op2monitor.update(module.op2monitor)
  for key in module.submods:
    submod = module.submods[key]
    _recursive_gather_op2monitor_helper(submod, op2monitor)
