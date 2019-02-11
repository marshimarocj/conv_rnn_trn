import json
import cPickle
import random
import enum
import sys
import os
import math
sys.path.append('../')

import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score

import framework.model.module
import framework.model.trntst
import framework.model.data
import framework.impl.encoder.conv_rnn
import framework.impl.encoder.pca

CONV_RNN = 'conv_rnn'
CONV = 'pca'
CELL = framework.impl.encoder.conv_rnn.CELL

class ModelConfig(framework.model.module.ModelConfig):
  def __init__(self):
    framework.model.module.ModelConfig.__init__(self)

    self.subcfgs[CONV_RNN] = framework.impl.encoder.conv_rnn.RNNConfig()
    self.subcfgs[CONV] = framework.impl.encoder.pca.Config()
    self.h = 5
    self.w = 5
    self.num_pos_class = 19
    self.dim_hiddens = []
    self.pool = 'mean'
    self.delay = 4
    self.norms = [False]

  def _assert(self):
    assert self.subcfgs[CONV_RNN].subcfgs[CELL].dim_input == self.subcfgs[CONV].dim_output
    assert self.subcfgs[CONV_RNN].num_step >= self.delay


def gen_cfg(**kwargs):
  cfg = ModelConfig()
  cfg.val_iter = -1
  cfg.monitor_iter = 50
  cfg.trn_batch_size = 32
  cfg.tst_batch_size = 32
  cfg.base_lr = 1e-4
  cfg.val_loss = False
  cfg.num_epoch = 10
  cfg.num_pos_class = kwargs['num_pos_class']
  cfg.h = kwargs['hw'][0]
  cfg.w = kwargs['hw'][1]
  cfg.dim_hiddens = kwargs['dim_hiddens']
  cfg.delay = kwargs['delay']

  conv_rnn_cfg = cfg.subcfgs[CONV_RNN]
  conv_rnn_cfg.num_step = kwargs['num_step']
  conv_rnn_cfg.cell = kwargs['cell']

  cell_cfg = conv_rnn_cfg.subcfgs[CELL]
  cell_cfg.dim_hidden = kwargs['cell_dim_hidden']
  cell_cfg.dim_input = kwargs['dim_embed']
  cell_cfg.conv_kernel = kwargs['conv_kernel']
  cell_cfg.keepout_prob = 0.5
  cell_cfg.keepin_prob = 0.5

  conv_cfg = cfg.subcfgs[CONV]
  conv_cfg.dim_ft = kwargs['dim_ft']
  conv_cfg.dim_output = kwargs['dim_embed']

  return cfg


class Model(framework.model.module.AbstractModel):
  name_scope = 'conv_rnn.Model'

  class InKey(enum.Enum):
    FT = 'fts'
    IS_TRN = 'is_training'
    LABEL = 'label'
    LABEL_MASK = 'label_mask'

  class OutKey(enum.Enum):
    LOGIT = 'logit'
    PREDICT = 'predict'

  def _set_submods(self):
    return {
      CONV_RNN: framework.impl.encoder.conv_rnn.Encoder(self._config.subcfgs[CONV_RNN]),
      CONV: framework.impl.encoder.pca.Encoder2D(self._config.subcfgs[CONV]),
    }

  def _add_input_in_mode(self, mode):
    conv_rnn_cfg = self._config.subcfgs[CONV_RNN]
    conv_cfg = self._config.subcfgs[CONV]
    with tf.variable_scope(self.name_scope):
      fts = tf.placeholder(
        tf.float32, shape=(None, conv_rnn_cfg.num_step, self._config.h, self._config.w, conv_cfg.dim_ft),
        name=self.InKey.FT.value)
      is_training = tf.placeholder(
        tf.bool, shape=(), name=self.InKey.IS_TRN.value)
      if mode == framework.model.module.Mode.TRN_VAL:
        label = tf.placeholder(
          tf.float32, shape=(None, conv_rnn_cfg.num_step, self._config.num_pos_class), name=self.InKey.LABEL.value)
        label_mask = tf.placeholder(
          tf.float32, shape=(None, conv_rnn_cfg.num_step, self._config.num_pos_class), name=self.InKey.LABEL_MASK.value)
        return {
          self.InKey.FT: fts,
          self.InKey.LABEL: label,
          self.InKey.LABEL_MASK: label_mask,
          self.InKey.IS_TRN: is_training,
        }
      else:
        return {
          self.InKey.FT: fts,
          self.InKey.IS_TRN: is_training,
        }

  def _build_parameter_graph(self):
    conv_rnn_cfg = self._config.subcfgs[CONV_RNN]
    self.fc_Ws = []
    self.fc_Bs = []
    with tf.variable_scope(self.name_scope):
      dim_inputs = [conv_rnn_cfg.subcfgs[CELL].dim_hidden] + self._config.dim_hiddens[:-1]
      dim_outputs = self._config.dim_hiddens
      i = 0
      for dim_input, dim_output in zip(dim_inputs, dim_outputs):
        fc_W = tf.contrib.framework.model_variable('fc_W_%d'%i,
          shape=(1, dim_input, dim_output), dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        self._weights.append(fc_W)
        fc_B = tf.contrib.framework.model_variable('fc_B_%d'%i,
          shape=(dim_output,), dtype=tf.float32,
          initializer=tf.constant_initializer(0.))
        self._weights.append(fc_B)
        self.fc_Ws.append(fc_W)
        self.fc_Bs.append(fc_B)
        dim_input = dim_output
        i += 1

      self.sigmoid_W = tf.contrib.framework.model_variable('sigmoid_W',
        shape=(1, self._config.dim_hiddens[-1], self._config.num_pos_class), dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer())
      self.sigmoid_B = tf.contrib.framework.model_variable('sigmoid_B',
        shape=(self._config.num_pos_class,), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
      self._weights.append(self.sigmoid_W)
      self._weights.append(self.sigmoid_B)

  def get_out_ops_in_mode(self, in_ops, mode):
    out_ops = self.submods[CONV].get_out_ops_in_mode({
      self.submods[CONV].InKey.FT: in_ops[self.InKey.FT],
      }, mode)

    with tf.variable_scope(self.name_scope):
      shape = tf.shape(in_ops[self.InKey.FT])
      init_state = tf.zeros((shape[0], self._config.h, self._config.w, self._config.subcfgs[CONV_RNN].subcfgs[CELL].dim_hidden))

    out_ops = self.submods[CONV_RNN].get_out_ops_in_mode({
      self.submods[CONV_RNN].InKey.FT: out_ops[self.submods[CONV].OutKey.EMBED],
      self.submods[CONV_RNN].InKey.IS_TRN: in_ops[self.InKey.IS_TRN],
      self.submods[CONV_RNN].InKey.INIT_STATE: init_state,
      }, mode)

    tst_outputs = out_ops[self.submods[CONV_RNN].OutKey.TST_OUTPUT] # (None, num_step, H, W, dim_hidden)
    if mode == framework.model.module.Mode.TRN_VAL:
      outputs = out_ops[self.submods[CONV_RNN].OutKey.OUTPUT]

    with tf.variable_scope(self.name_scope):
      # spatial pooling
      if self._config.pool == 'mean':
        tst_outputs = tf.reduce_mean(tf.reduce_mean(tst_outputs, axis=2), axis=2) # (None, num_step, dim_hidden)

      for fc_W, fc_B in zip(self.fc_Ws, self.fc_Bs):
        tst_outputs = tf.nn.conv1d(tst_outputs, fc_W, 1, 'VALID')
        tst_outputs = tf.nn.bias_add(tst_outputs, fc_B)
        tst_outputs = tf.nn.relu(tst_outputs)
      tst_predicts = tf.nn.conv1d(tst_outputs, self.sigmoid_W, 1, 'VALID')
      tst_predicts = tf.nn.bias_add(tst_predicts, self.sigmoid_B)
      tst_predicts = tf.nn.sigmoid(tst_predicts) # (None, num_step, num_pos_class)

      if mode == framework.model.module.Mode.TRN_VAL:
        # spatial pooling
        if self._config.pool == 'mean':
          outputs = tf.reduce_mean(tf.reduce_mean(outputs, 2), 2) # (None, num_step, dim_hidden)
        elif self._config.pool == 'max':
          outputs = tf.reduce_max(tf.reduce_max(outputs, 2), 2)
        for fc_W, fc_B in zip(self.fc_Ws, self.fc_Bs):
          outputs = tf.nn.conv1d(outputs, fc_W, 1, 'VALID')
          outputs = tf.nn.bias_add(outputs, fc_B)
          outputs = tf.nn.relu(outputs)
        logits = tf.nn.conv1d(outputs, self.sigmoid_W, 1, 'VALID')
        logits = tf.nn.bias_add(logits, self.sigmoid_B)

        return {
          self.OutKey.LOGIT: logits,
          self.OutKey.PREDICT: tst_predicts,
        }
      else:
        return {
          self.OutKey.PREDICT: tst_predicts,
        }

  def _add_loss(self):
    with tf.variable_scope(self.name_scope):
      labels = self._inputs[self.InKey.LABEL]
      label_masks = self._inputs[self.InKey.LABEL_MASK]
      logits = self._outputs[self.OutKey.LOGIT]

      labels = tf.reshape(labels, (-1, self._config.num_pos_class))
      label_masks = tf.reshape(label_masks, (-1, self._config.num_pos_class))
      logits = tf.reshape(logits, (-1, self._config.num_pos_class))
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)
      loss = tf.reduce_sum(loss * label_masks) / tf.reduce_sum(label_masks)

    return loss

  def op_in_val(self, **kwargs):
    return {
      self.OutKey.PREDICT: self._outputs[self.OutKey.PREDICT],
    }

  def op_in_tst(self, **kwargs):
    return {
      self.OutKey.PREDICT: self._outputs[self.OutKey.PREDICT],
    }


class TrnTst(framework.model.trntst.TrnTst):
  def feed_data_and_run_loss_op_in_val(self, data, sess):
    op_dict = self.model.op_in_val()

    feed_dict = self._construct_feed_dict_in_val(data)
    loss = sess.run(op_dict[self.model.DefaultKey.LOSS], feed_dict=feed_dict)

    return loss

  def _construct_feed_dict_in_trn(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.IS_TRN]: True,
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.LABEL_MASK]: data['masks'],
    }

  def _construct_feed_dict_in_val(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
      self.model.inputs[self.model.InKey.LABEL]: data['labels'],
      self.model.inputs[self.model.InKey.LABEL_MASK]: data['masks'],
    }

  def _construct_feed_dict_in_tst(self, data):
    return {
      self.model.inputs[self.model.InKey.FT]: data['fts'],
      self.model.inputs[self.model.InKey.IS_TRN]: False,
    }

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    label2lid = tst_reader.label2lid
    lid2label = {}
    for label in label2lid:
      lid2label[label2lid[label]] = label
    num_label = len(label2lid)

    op_dict = self.model.op_in_val()
    tst_batch_size = self.model_cfg.tst_batch_size

    all_predicts = []
    all_labels = []
    for data in tst_reader.yield_tst_batch(tst_batch_size):
      masks = data['masks']
      labels = data['labels']
      feed_dict = self._construct_feed_dict_in_val(data)
      predicts = sess.run(op_dict[self.model.OutKey.PREDICT], feed_dict=feed_dict)
      for predict, mask, label in zip(predicts, masks, labels):
        num_step = mask.shape[0]
        for i in range(num_step):
          if np.sum(mask[i]) != num_label:
            continue
          all_predicts.append(predict[i])
          all_labels.append(label[i])
    all_predicts = np.array(all_predicts)
    all_labels = np.array(all_labels)

    mAP = 0.
    for l in range(num_label):
      score = average_precision_score(all_labels[:, l], all_predicts[:, l])
      metrics[lid2label[l]] = score
      mAP += score
    metrics['mAP'] = mAP / num_label

    all_predicts = all_predicts.reshape((-1,))
    all_labels = all_labels.reshape((-1,))
    score = average_precision_score(all_labels, all_predicts)
    metrics['GAP'] = score

  def predict_in_tst(self, sess, tst_reader, predict_file):
    op_dict = self.model.op_in_val()
    tst_batch_size = self.model_cfg.tst_batch_size

    all_predicts = []
    all_labels = []
    for data in tst_reader.yield_tst_batch(tst_batch_size):
      masks = data['masks']
      labels = data['labels']
      feed_dict = self._construct_feed_dict_in_tst(data)
      predicts = sess.run(op_dict[self.model.OutKey.PREDICT], feed_dict=feed_dict)
      for predict, mask, label in zip(predicts, masks, labels):
        num_step = mask.shape[0]
        for i in range(num_step):
          if np.sum(mask[i]) != predict[i].shape[-1]:
            continue
          all_predicts.append(predict[i])
          all_labels.append(label[i])
    all_predicts = np.array(all_predicts)
    all_labels = np.array(all_labels)

    np.savez_compressed(predict_file, predicts=all_predicts, labels=all_labels)


class PathCfg(framework.model.trntst.PathCfg):
  def __init__(self):
    framework.model.trntst.PathCfg.__init__(self)

    self.trn_dirs = []
    self.val_dir = ''
    self.label2lid_file = ''

    self.output_dir = ''
    self.log_file = ''


class Reader(framework.model.data.Reader):
  def __init__(self, label2lid_file, data_dirs, num_step, delay, shuffle=False):
    self.label2lid = {}
    self.shuffle = shuffle
    self.num_step = num_step
    self.delay = delay

    self.fts = []
    self.labels = []
    self.masks = []
    self.props_names = []

    with open(label2lid_file) as f:
      self.label2lid = json.load(f)

    for data_dir in data_dirs:
      names = os.listdir(data_dir)
      names = sorted(names)
      names = names[:1]

      options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
      for name in names:
        chunk_file = os.path.join(data_dir, name)
        record_iterator = tf.python_io.tf_record_iterator(path=chunk_file, options=options)
        record_iterator.next()
        for record in record_iterator:
          ft, label, label_mask, eid, video = parse_record(record)
          self.fts.append(ft)
          self.labels.append(label)
          self.masks.append(label_mask)
          self.props_names.append('%s_%d'%(video, eid))

    self.total = len(self.fts)
    self.idxs = range(self.total)

  def reset(self):
    if self.shuffle:
      random.shuffle(self.idxs)

  def num_record(self):
    return self.total # pseudo

  def yield_trn_batch(self, batch_size):
    batch_fts = []
    batch_labels = []
    batch_masks = []
    for idx in self.idxs:
      ft = self.fts[idx]
      label = self.labels[idx]
      label_mask = self.masks[idx]
      num = ft.shape[0]
      i = 0
      while i < num:
        if i == 0:
          delay = 0
        else:
          delay = self.delay
        pad_ft, pad_label, pad_label_mask = pad_data(ft, label, label_mask, i, self.num_step, delay)
        batch_fts.append(pad_ft)
        batch_labels.append(pad_label)
        batch_masks.append(pad_label_mask)
        i += self.num_step - self.delay

        if len(batch_fts) == batch_size:
          yield {
            'fts': batch_fts,
            'labels': batch_labels,
            'masks': batch_masks,
          }
          batch_fts = []
          batch_labels = []
          batch_masks = []
    if len(batch_fts) > 0:
      yield {
        'fts': batch_fts,
        'labels': batch_labels,
        'masks': batch_masks,
      }

  def yield_tst_batch(self, batch_size):
    for data in self.yield_trn_batch(batch_size):
      yield data


class TstReader(framework.model.data.Reader):
  def __init__(self, label2lid_file, data_dir, num_step, delay):
    self.label2lid = {}
    self.num_step = num_step
    self.delay = delay

    self.fts = []
    self.props_names = []

    with open(label2lid_file) as f:
      self.label2lid = json.load(f)

    self.data_dir = data_dir
    self.names = os.listdir(self.data_dir)
    self.names = sorted(self.names)

  def num_record(self):
    return self.total

  def yield_tst_batch(self, batch_size):
    batch_fts = []
    batch_steps = []
    batch_masks = []
    batch_props_names = []

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    for name in self.names:
      print name
      chunk_file = os.path.join(self.data_dir, name)
      record_iterator = tf.python_io.tf_record_iterator(path=chunk_file, options=options)
      record_iterator.next()
      for record in record_iterator:
        ft, eid, video = parse_tst_record(record)
        props_name = '%s_%d'%(video, eid)

        num = ft.shape[0]
        i = 0
        while i < num:
          pad_ft, pad_mask = pad_tst_data(ft, i, self.num_step)
          step = range(i, i + self.num_step)

          batch_fts.append(pad_ft)
          batch_steps.append(step)
          batch_masks.append(pad_mask)
          batch_props_names.append(props_name)
          i += self.num_step - self.delay

          if len(batch_fts) == batch_size:
            yield {
              'fts': batch_fts,
              'steps': batch_steps,
              'masks': batch_masks,
              'props_names': batch_props_names,
            }
            batch_fts = []
            batch_steps = []
            batch_masks = []
            batch_props_names = []
    if len(batch_fts) > 0:
      yield {
        'fts': batch_fts,
        'steps': batch_steps,
        'masks': batch_masks,
        'props_names': batch_props_names,
      }


def parse_record(record):
  example = tf.train.Example()
  example.ParseFromString(record)

  ft = example.features.feature['fts'].bytes_list.value[0]
  shape = example.features.feature['fts.shape'].int64_list.value
  shape = [d for d in shape]
  ft = np.fromstring(ft, dtype=np.float32)
  ft = ft.reshape(shape)

  label = example.features.feature['labels'].bytes_list.value[0]
  shape = example.features.feature['labels.shape'].int64_list.value
  shape = [d for d in shape]
  label = np.fromstring(label, dtype=np.float32)
  label = label.reshape(shape)

  label_mask = example.features.feature['label_masks'].bytes_list.value[0]
  shape = example.features.feature['label_masks.shape'].int64_list.value
  shape = [d for d in shape]
  label_mask = np.fromstring(label_mask, dtype=np.float32)
  label_mask = label_mask.reshape(shape)

  eid = example.features.feature['eid'].int64_list.value[0]
  video = ''.join([d for d in example.features.feature['video'].bytes_list.value])

  return ft, label, label_mask, eid, video


def pad_data(fts, labels, label_masks, start, num_step, delay):
  ft = fts[start:start + num_step]
  label = labels[start:start + num_step]
  label_mask = label_masks[start:start + num_step]
  if ft.shape[0] < num_step:
    num_fill = num_step - ft.shape[0]
    ft_fill = np.zeros((num_fill,) + ft.shape[1:], dtype=np.float32)
    ft = np.concatenate([ft, ft_fill], axis=0)

    label_fill = np.zeros((num_fill,) + label.shape[1:], dtype=np.float32)
    label = np.concatenate([label, label_fill], axis=0)

    label_mask_fill = np.zeros((num_fill,) + label_mask.shape[1:], dtype=np.float32)
    label_mask = np.concatenate([label_mask, label_mask_fill], axis=0)

  if start != 0:
    label_mask[:delay-1] = 0.

  return ft, label, label_mask


def parse_tst_record(record):
  example = tf.train.Example()
  example.ParseFromString(record)

  ft = example.features.feature['fts'].bytes_list.value[0]
  shape = example.features.feature['fts.shape'].int64_list.value
  shape = [d for d in shape]
  ft = np.fromstring(ft, dtype=np.float32)
  ft = ft.reshape(shape)

  eid = example.features.feature['eid'].int64_list.value[0]
  video = ''.join([d for d in example.features.feature['video'].bytes_list.value])

  return ft, eid, video


def pad_tst_data(fts, start, num_step):
  ft = fts[start:start + num_step]
  mask = np.zeros((num_step,))
  mask[:ft.shape[0]] = 1.
  if ft.shape[0] < num_step:
    num_fill = num_step - ft.shape[0]
    ft_fill = np.zeros((num_fill,) + ft.shape[1:], dtype=np.float32)
    ft = np.concatenate([ft, ft_fill], axis=0)

  return ft, mask
