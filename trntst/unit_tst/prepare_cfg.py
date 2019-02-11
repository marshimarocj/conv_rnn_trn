import os
import sys
import json
sys.path.append('../')

import model.conv_rnn


def prepare_conv_rnn():
  root_dir = '/home/jiac/data_8t/actev' # aws1
  props_types = ['gt']
  ft_name = 'i3d_rgb'
  trn_data_dirs = [
    os.path.join(root_dir, 'compiled_data_loop', props_type, ft_name, 'trn') for props_type in props_types
  ]
  val_data_dir = os.path.join(root_dir, 'compiled_data_loop', 'heu_gt_obj_props', ft_name, 'val')
  expr_dir = os.path.join(root_dir, 'expr_deliver', 'conv_rnn')

  params = {
    'dim_hiddens': [256],
    'dim_embed': 256,
    'cell_dim_hidden': 256,
    'num_step': 64,
    'num_pos_class': 19,
    'conv_kernel': [3,3],
    'cell': 'gru',
    'hw': [7, 7],
    'dim_ft': 1024,
    'delay': 4,
  }

  outprefix = '%s/%s.%s.%d.%s.%s'%(
    expr_dir, ft_name, '_'.join([str(d) for d in params['dim_hiddens']]), 
    params['cell_dim_hidden'], 
    'x'.join([str(d) for d in params['conv_kernel']]),
    '-'.join(props_types), 
  )
  model_cfg_file = '%s.model.json'%outprefix
  cfg = model.conv_rnn.gen_cfg(**params)
  cfg.save(model_cfg_file)

  path_cfg = {
    'trn_dirs': trn_data_dirs,
    'val_dir': val_data_dir,
    'output_dir': outprefix,
    'label2lid_file': os.path.join(root_dir, 'lst', 'label2lid_%d.json'%params['num_pos_class'])
  }
  path_cfg_file = '%s.path.json'%outprefix
  with open(path_cfg_file, 'w') as fout:
    json.dump(path_cfg, fout, indent=2)

  if not os.path.exists(outprefix):
    os.mkdir(outprefix)


if __name__ == '__main__':
  prepare_conv_rnn()
