import argparse
import sys
import os
import datetime
import cPickle
sys.path.append('../')

import numpy as np

import model.conv_rnn
import common

RNN = model.conv_rnn.CONV_RNN


def build_parser():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument('model_cfg_file', description='''
 configuration file of model and train paramters
    ''', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('path_cfg_file', description='''
configuration file of data and experiment directories
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--is_train', dest='is_train', type=int, default=True,
    description='''
1 for train mode and 0 for test mode
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # only in tst
  parser.add_argument('--best_epoch', dest='best_epoch', type=int, default=0,
    description='''
option only in tst mode, the epoch used in tst
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--data_dir', default='', description='''
option only in tst mode, the data_dir used in tst
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--out_name', default='', description='''
option only in tst mode, the output file name in tst
    ''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  return parser


def load_and_fill_model_cfg(model_cfg_file, path_cfg):
  model_cfg = model.conv_rnn.ModelConfig()
  model_cfg.load(model_cfg_file)

  return model_cfg


if __name__ == '__main__':
  parser = build_parser()
  opts = parser.parse_args()

  path_cfg = model.conv_rnn.PathCfg()
  common.gen_dir_struct_info(path_cfg, opts.path_cfg_file)
  model_cfg = load_and_fill_model_cfg(opts.model_cfg_file, path_cfg)

  m = model.conv_rnn.Model(model_cfg)

  if opts.is_train:
    trntst = model.conv_rnn.TrnTst(model_cfg, path_cfg, m)

    trn_reader = model.conv_rnn.Reader(
      path_cfg.label2lid_file, path_cfg.trn_dirs, model_cfg.subcfgs[RNN].num_step, model_cfg.delay, shuffle=True)
    val_reader = model.conv_rnn.Reader(
      path_cfg.label2lid_file, [path_cfg.val_dir], model_cfg.subcfgs[RNN].num_step, model_cfg.delay, shuffle=False)
    if path_cfg.model_file != '':
      trntst.train(trn_reader, val_reader, memory_fraction=1., resume=True)
    else:
      trntst.train(trn_reader, val_reader, memory_fraction=1.)
  else:
    path_cfg.model_file = os.path.join(path_cfg.model_dir, 'epoch-%d'%opts.best_epoch)
    path_cfg.log_file = None
    path_cfg.predict_file = os.path.join(path_cfg.output_dir, 'pred', '%s.npz'%opts.out_name)
    data_dir = path_cfg.val_dir if opts.data_dir == '' else opts.data_dir

    trntst = model.conv_rnn.TrnTst(model_cfg, path_cfg, m)
    tst_reader = model.conv_rnn.Reader(
      path_cfg.label2lid_file, [data_dir], model_cfg.subcfgs[RNN].num_step, model_cfg.delay, shuffle=False)
    trntst.test(tst_reader, memory_fraction=1.)
