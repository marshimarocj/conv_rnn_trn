import os
import random
from collections import deque

import tensorflow as tf


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def meta_record(num_record):
  meta = tf.train.Example(features=tf.train.Features(feature={
    'num_record': int64_feature([num_record]),
  }))
  return meta
