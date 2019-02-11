import os
import argparse
import json

import numpy as np
import tensorflow as tf

import framework.util.io
from diva_common.structure.annotation import *


'''func
'''
def bilinear_interpolate_align(im, h, w):
  oh, ow, _ = im.shape
  oh = float(oh)
  ow = float(ow)
  y = np.arange(h) * (oh-1) / (h-1) # align start and end point, [start, end]
  x = np.arange(w) * (ow-1) / (w-1)
  y, x = np.meshgrid(y, x, indexing='ij')
  y = y.reshape((-1,))
  x = x.reshape((-1,))

  x0 = np.floor(x).astype(int)
  x0[-1] -= 1
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y0[-1] -= 1
  y1 = y0 + 1

  Ia = im[y0, x0]
  Ib = im[y1, x0]
  Ic = im[y0, x1]
  Id = im[y1, x1]

  wa = (x1-x) * (y1-y)
  wb = (x1-x) * (y-y0)
  wc = (x-x0) * (y1-y)
  wd = (x-x0) * (y-y0)

  wa = np.expand_dims(wa, 1)
  wb = np.expand_dims(wb, 1)
  wc = np.expand_dims(wc, 1)
  wd = np.expand_dims(wd, 1)

  I = wa*Ia + wb*Ib + wc*Ic + wd*Id
  return I.reshape((h, w, -1))


def norm_ft(ft_file, dst_h=7, dst_w=7):
  data = np.load(ft_file)
  fts = data['feat']
  if len(fts.shape) == 0:
    return None

  num, t, h, w, c = fts.shape
  if h != dst_h or w != dst_w:
    resize_fts = []
    for segment_ft in fts:
      segment = []
      for ft in segment_ft:
        resize_ft = bilinear_interpolate(ft, h, w)
        segment.append(resize_ft)
      resize_fts.append(segment)
    fts = np.array(resize_fts, dtype=np.float32)

  dim_ft = fts.shape[-1]

  mean_ft = np.zeros((t+t/2*(fts.shape[0]-1), h, w, dim_ft), dtype=np.float32)
  for j, ft in enumerate(fts):
    mean_ft[j*t/2:j*t/2+t] += ft
  mean_ft[t/2:-t/2] /= 2.

  return mean_ft


'''expr
'''
def prepare_data():
  root_dir = '/home/jiac/data_8t/actev'
  lst_file = os.path.join(root_dir, 'official_data', 'VIRAT-V1_JSON_train-leaderboard_drop4_20180614', 'file-index.json')

  props_type = 'gt'
  split = 'trn'
  ft_name = 'i3d_rgb'
  chunk = 0

  with open(lst_file) as f:
    data = json.load(f)
  videos = [video.split('.')[0] for video in data.keys()]
  videos = videos[:1] # only one video for illustration

  actv_dir = os.path.join(root_dir, 'proposal_annotations', props_type)
  ft_dir = os.path.join(root_dir, 'features', props_type, ft_name)
  label_dir = os.path.join(root_dir, 'compiled_label', props_type)
  out_dir = os.path.join(root_dir, 'expr_deliver', ft_name)

  records = []
  for video in videos:
    print video
    actv_file = os.path.join(actv_dir, video, 'actv_id_type.pkl')
    actv = ActvIdType()
    actv.load(actv_file)
    actv.index()

    label_file = os.path.join(label_dir, '%s.pkl'%video)
    with open(label_file) as f:
      eid2label_and_mask = cPickle.load(f)

    for eid in actv.roots:
      ft_file = os.path.join(ft_dir, '%s_%d.npz'%(video, eid))
      if not os.path.exists(ft_file):
        continue
      fts = norm_ft(ft_file)
      if fts is None:
        continue

      labels = eid2label_and_mask[eid]['labels']
      label_masks = eid2label_and_mask[eid]['label_masks']

      num = min(fts.shape[0], labels.shape[0])
      fts = fts[:num].astype(np.float32)
      labels = labels[:num].astype(np.float32)
      label_masks = label_masks[:num].astype(np.float32)

      example = tf.train.Example(features=tf.train.Features(feature={
        'fts': framework.util.io.bytes_feature([fts.tostring()]),
        'fts.shape': framework.util.io.int64_feature(fts.shape),
        'labels': framework.util.io.bytes_feature([labels.tostring()]),
        'labels.shape': framework.util.io.int64_feature(labels.shape),
        'label_masks': framework.util.io.bytes_feature([label_masks.tostring()]),
        'label_masks.shape': framework.util.io.int64_feature(label_masks.shape),
        'eid': framework.util.io.int64_feature([eid]),
        'video': framework.util.io.bytes_feature([video.encode('ascii', 'ignore')]),
      }))
      records.append(example)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    out_file = os.path.join(out_dir, split, '%d.tfrecord'%chunk)
    with tf.python_io.TFRecordWriter(out_file, options=options) as writer:
      meta_record = framework.util.io.meta_record(len(records))
      writer.write(meta_record.SerializeToString())
      for record in records:
        writer.write(record.SerializeToString())
    print 'chunk:', chunk


if __name__ == '__main__':
  prepare_data()
