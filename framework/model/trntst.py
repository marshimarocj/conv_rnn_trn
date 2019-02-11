import os
import sys
import time
import cPickle
import collections
import json
import pprint
import logging
sys.path.append('../')

import tensorflow as tf
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader


class PathCfg(object):
  def __init__(self):
    self.log_dir = ''
    self.model_dir = ''

    self.log_file = ''
    self.model_file = ''
    self.predict_file = ''

  def load(self, file):
    data = json.load(open(file))
    for key in data:
      setattr(self, key, data[key])


class TrnTst(object):
  def __init__(self, model_cfg, path_cfg, model, debug=False):
    self._model_cfg = None
    self._path_cfg = None
    self._model = None
    self._logger = None
    self._debug = debug

    self._model_cfg = model_cfg
    self._path_cfg = path_cfg
    self._model = model

    self._logger = set_logger('TrnTst', path_cfg.log_file)

  @property
  def model_cfg(self):
    return self._model_cfg

  @property
  def path_cfg(self):
    return self._path_cfg

  @property
  def model(self):
    return self._model

  ######################################
  # functions to customize 
  ######################################
  def _construct_feed_dict_in_trn(self, data):
    raise NotImplementedError("""please customize _construct_feed_dict_in_trn""")

  def feed_data_and_run_loss_op_in_val(self, data, sess):
    """
    return loss value
    """
    raise NotImplementedError("""please customize feed_data_and_run_loss_op_in_val""")

  def predict_and_eval_in_val(self, sess, tst_reader, metrics):
    """
    add eval result to metrics dictionary, key is metric name, val is metric value
    """
    raise NotImplementedError("""please customize predict_and_eval_in_val""")

  def predict_in_tst(self, sess, tst_reader, predict_file):
    """
    write predict result to predict_file
    """
    raise NotImplementedError("""please customize predict_in_tst""")

  def customize_func_after_each_epoch(self, epoch):
    pass

  ######################################
  # boilerpipe functions
  ######################################
  def feed_data_and_trn(self, data, sess):
    op_dict = self.model.op_in_trn()

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(
      [op_dict[self.model.DefaultKey.LOSS]] + op_dict[self.model.DefaultKey.TRAIN],
      feed_dict=feed_dict)
    return out[0]

  def feed_data_and_monitor_in_trn(self, data, sess, step):
    op2monitor = self.model.op2monitor
    names = op2monitor.keys()
    ops = op2monitor.values()

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(ops, feed_dict=feed_dict)
    for name, val in zip(names, out):
      self._logger.info('(step %d) monitor "%s":%s', step, name, pprint.pformat(val))

  def feed_data_and_summary(self, data, sess):
    summary_op = self.model.summary_op

    feed_dict = self._construct_feed_dict_in_trn(data)
    out = sess.run(summary_op, feed_dict=feed_dict)

    return out

  def _iterate_epoch(self,
      sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch):
    trn_batch_size = self.model_cfg.trn_batch_size
    trn_time = 0.
    trn_reader.reset()
    avg_loss = 0.
    cnt = 0
    for data in trn_reader.yield_trn_batch(trn_batch_size):
      tic = time.time()
      loss = self.feed_data_and_trn(data, sess)
      toc = time.time()
      trn_time += toc - tic

      avg_loss += loss
      step += 1
      cnt += 1

      if self.model_cfg.monitor_iter > 0 and step % self.model_cfg.monitor_iter == 0:
        self.feed_data_and_monitor_in_trn(data, sess, step)

      if self.model_cfg.val_iter > 0 and step % self.model_cfg.val_iter == 0:
        tic = time.time()
        metrics = self._validation(sess, tst_reader)
        toc = time.time()
        val_time = toc - tic

        self._logger.info('step (%d/%d)', step, total_step)
        self._logger.info('%f s for trn', trn_time)
        self._logger.info('%f s for val', val_time)
        trn_time = 0.
        rollout_time = 0.
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])

    summarystr = self.feed_data_and_summary(data, sess)
    summarywriter.add_summary(summarystr, step)
    self.model.saver.save(
      sess, os.path.join(self.path_cfg.model_dir, 'epoch'), global_step=epoch)

    avg_loss /= cnt
    return step, avg_loss

  def _validation(self, sess, tst_reader):
    metrics = collections.OrderedDict()
    batch_size = self.model_cfg.tst_batch_size

    # loss on validation
    if self.model_cfg.val_loss:
      iter_num = 0
      avg_loss = 0.
      for data in tst_reader.yield_val_batch(batch_size):
        loss = self.feed_data_and_run_loss_op_in_val(data, sess)
        avg_loss += loss
        iter_num += 1
        # print loss

      avg_loss /= iter_num
      metrics['loss'] = avg_loss

    self.predict_and_eval_in_val(sess, tst_reader, metrics)

    return metrics

  def train(self, trn_reader, tst_reader, memory_fraction=1.0, resume=False):
    batch_size = self.model_cfg.trn_batch_size
    batches_per_epoch = (trn_reader.num_record() + batch_size - 1) / batch_size
    total_step = batches_per_epoch * self.model_cfg.num_epoch

    decay_boundarys = []
    step = 0
    trn_tst_graph = self.model.build_trn_tst_graph(decay_boundarys=decay_boundarys, step=step)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    configProto = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    with tf.Session(graph=trn_tst_graph, config=configProto) as sess:
      sess.run(self.model.init_op)
      base_epoch = 0
      if resume:
        self.model.saver.restore(sess, self.path_cfg.model_file)
        name = os.path.basename(self.path_cfg.model_file)
        data = name.split('-')
        try:
          base_epoch = int(data[-1]) + 1
        except:
          pass
      summarywriter = tf.summary.FileWriter(self.path_cfg.log_dir, graph=sess.graph)

      # round 0, just for quick checking
      metrics = self._validation(sess, tst_reader)
      self._logger.info('step (%d)', 0)
      for key in metrics:
        self._logger.info('%s:%.4f', key, metrics[key])

      for epoch in xrange(base_epoch, self.model_cfg.num_epoch):
        step, avg_loss = self._iterate_epoch(
          sess, trn_reader, tst_reader, summarywriter, step, total_step, epoch)

        metrics = self._validation(sess, tst_reader)
        metrics['epoch'] = epoch
        metrics['train_loss'] = avg_loss

        self._logger.info('epoch (%d/%d)', epoch, self.model_cfg.num_epoch)
        for key in metrics:
          self._logger.info('%s:%.4f', key, metrics[key])
        val_log_file = os.path.join(self.path_cfg.log_dir, 'val_metrics.%d.json'%epoch)
        with open(val_log_file, 'w') as fout:
          json.dump(metrics, fout, indent=2)

        self.customize_func_after_each_epoch(epoch)

  def test(self, tst_reader, memory_fraction=1.0):
    tst_graph = self.model.build_tst_graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_fraction)
    config_proto = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    with tf.Session(graph=tst_graph, config=config_proto) as sess:
      sess.run(self.model.init_op)
      if self.path_cfg.model_file is not None:
        self.model.saver.restore(sess, self.path_cfg.model_file)

      self.predict_in_tst(sess, tst_reader, self.path_cfg.predict_file)


def set_logger(name, log_path=None):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)

  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
  logger.addHandler(console)

  if log_path is not None:
    if os.path.exists(log_path):
      os.remove(log_path)

    logfile = logging.FileHandler(log_path)
    logfile.setLevel(logging.INFO)
    logfile.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(logfile)

  return logger