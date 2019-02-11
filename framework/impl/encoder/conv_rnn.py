import enum
import math
import sys

import tensorflow as tf
from tensorflow.python.util import nest

import framework.model.module
import framework.impl.conv_cell

CELL = 'cell'


class RNNConfig(framework.model.module.ModuleConfig):
  def __init__(self):
    framework.model.module.ModuleConfig.__init__(self)

    self.subcfgs[CELL] = framework.impl.conv_cell.CellConfig()
    self.cell = 'gru'

    self.num_step = 10

  def _assert(self):
    pass


class Encoder(framework.model.module.AbstractModule):
  name_scope = 'conv_rnn.Encoder'

  class InKey(enum.Enum):
    FT = 'ft' # (None, num_step, H, W, dim_ft)
    IS_TRN = 'is_training'
    INIT_STATE = 'init_state'

  class OutKey(enum.Enum):
    OUTPUT = 'output'
    TST_OUTPUT = 'tst_output'

  def _set_submods(self):
    if self._config.cell == 'gru':
      return {
        CELL: framework.impl.conv_cell.GRUCell(self._config.subcfgs[CELL]),
      }

  def _build_parameter_graph(self):
    pass

  def _steps(self, fts, state, is_training):
    state_size = self.submods[CELL].state_size
    state = [state for _ in nest.flatten(state_size)]
    state = nest.pack_sequence_as(state_size, state)
    
    outputs = []
    for i in range(self._config.num_step):
      ft = fts[:, i] # (None, H, W, dim_ft)
      out = self.submods[CELL].get_out_ops_in_mode({
        self.submods[CELL].InKey.INPUT: ft,
        self.submods[CELL].InKey.STATE: state,
        self.submods[CELL].InKey.IS_TRN: is_training,
        }, None)
      output = out[self.submods[CELL].OutKey.OUTPUT] # (None, H, W, dim_hidden)
      state = out[self.submods[CELL].OutKey.STATE]
      outputs.append(output)
    outputs = tf.stack(outputs, axis=0)
    outputs = tf.transpose(outputs, perm=[1, 0, 2, 3, 4]) # (None, num_step, H, W, dim_hidden)
    return outputs

  def get_out_ops_in_mode(self, in_ops, mode):
    with tf.variable_scope(self.name_scope):
      outputs = self._steps(in_ops[self.InKey.FT], in_ops[self.InKey.INIT_STATE], in_ops[self.InKey.IS_TRN])
      if mode == framework.model.module.Mode.TRN_VAL:
        return {
          self.OutKey.OUTPUT: outputs,
          self.OutKey.TST_OUTPUT: outputs,
        }
      else:
        return {
          self.OutKey.OUTPUT: outputs,
          self.OutKey.TST_OUTPUT: outputs,
        }
