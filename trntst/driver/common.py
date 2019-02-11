import os
import datetime


def gen_dir_struct_info(path_cfg, path_cfg_file):
  path_cfg.load(path_cfg_file)

  output_dir = path_cfg.output_dir

  log_dir = os.path.join(output_dir, 'log')
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  model_dir = os.path.join(output_dir, 'model')
  if not os.path.exists(model_dir): 
    os.makedirs(model_dir)
  predict_dir = os.path.join(output_dir, 'pred')
  if not os.path.exists(predict_dir): 
    os.makedirs(predict_dir)

  path_cfg.log_dir = log_dir
  path_cfg.model_dir = model_dir

  timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  path_cfg.log_file = os.path.join(log_dir, 'log-' + timestamp)
  path_cfg.val_metric_file = os.path.join(log_dir, 'val_metrics.pkl')

  return path_cfg
