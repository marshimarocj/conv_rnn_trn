import os
import argparse
import json

import numpy as np
import tensorflow as tf

import framework.util.io


'''func
'''


'''expr
'''
def prepare_data():
  root_dir = '/home/jiac/data_8t/actev'
  lst_file = os.path.join(root_dir, 'official_data', 'VIRAT-V1_JSON_train-leaderboard_drop4_20180614', 'file-index.json')

  props_type = 'gt'
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
