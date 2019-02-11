import cPickle
import copy


# mapping to actv_id_meta.pkl
class ActvIdType(object):
  def __init__(self):
    self.eid2event_meta = {}

    # data structure for train and test
    self.roots = set()
    self.parent2children = {}

  def load(self, file):
    with open(file) as f:
      data = cPickle.load(f)
    for eid in data:
      meta = EventMeta()
      meta.eid = eid
      meta.parse(data[eid])
      self.eid2event_meta[eid] = meta

  def dump(self,file):
    data_to_dump = {}
    for eid, event_meta in self.eid2event_meta.iteritems():
      data_to_dump[eid] = event_meta.dump()
    with open(file,"w") as f:
      cPickle.dump(data_to_dump, f)

  def index(self):
    self.roots = set()
    self.parent2children = {}

    for eid in self.eid2event_meta:
      event_meta = self.eid2event_meta[eid]
      if eid == event_meta.parent:
        self.roots.add(eid)
      else:
        if event_meta.parent not in self.parent2children:
          self.parent2children[event_meta.parent] = []
        self.parent2children[event_meta.parent].append(eid)


class EventMeta(object):
  def __init__(self):
    self.eid = -1
    self.event = ''
    self.start_frame = -1
    self.end_frame = -1 # inclusive
    self.oid2object_meta = {} # object bboxs
    self.frame2bbx = {} # event bbox
    self.format = 'x1y1x2y2'

    # data structure for train and test
    self.parent = -1
    self.parent_offset_start = -1.
    self.parent_offset_end = -1.
    self.event_begin = -1
    self.event_end = -1

    # optional
    self.sigmoid_predicts = [] # used in prediction
    self.softmax_predicts = [] # used in prediction

  def parse(self, data):
    self.event = data['event_type']
    self.start_frame = data['start_frame']
    self.end_frame = data['end_frame'] # inclusive
    self.frame2bbx = data['trajectory']

    self.parent = data['parent']
    self.parent_offset_start = data['par_offset_st']
    self.parent_offset_end = data['par_offset_end']
    self.event_begin = data['event_begin']
    self.event_end = data['event_end']

    objects = data['objects']
    for oid in objects:
      meta = ObjectMeta()
      meta.parse(objects[oid])
      self.oid2object_meta[oid] = meta

    # optional
    if 'sigmoid_predicts' in data:
      self.sigmoid_predicts = data['sigmoid_predicts']
    if 'softmax_predicts' in data:
      self.softmax_predicts = data['softmax_predicts']

  def dump(self):
    ret_dict = {
      'start_frame': self.start_frame,
      'end_frame': self.end_frame,
      'event_type': self.event,
      'objects': {},
      'trajectory': self.frame2bbx,

      'parent': self.parent,
      'par_offset_st': self.parent_offset_start,
      'par_offset_end': self.parent_offset_end,
      'event_begin': self.event_begin,
      'event_end': self.event_end,

      'sigmoid_predicts': self.sigmoid_predicts,
      'softmax_predicts': self.softmax_predicts,
    }
    for oid in self.oid2object_meta:
      meta = self.oid2object_meta[oid]
      ret_dict['objects'][oid] = meta.dump()

    return ret_dict

  def generate_child(self, eid, start_frame, end_frame):
    child = EventMeta()
    child.event = self.event
    child.parent = self.eid
    child.eid = eid
    child.format = 'x1y1x2y2'

    child.start_frame = max(self.start_frame, start_frame)
    child.end_frame = min(self.end_frame, end_frame)
    child.event_begin = max(self.event_begin, start_frame)
    child.event_end = min(self.event_end, end_frame)

    parent_duration = float(self.end_frame - self.start_frame)
    child.parent_offset_start = (child.start_frame - self.start_frame) / parent_duration
    child.parent_offset_end = (child.end_frame - self.start_frame) / parent_duration

    return child

  # used in generating nist official output
  def cast_object_bbx_xywh(self):
    for oid in self.oid2object_meta:
      object_meta = self.oid2object_meta[oid]
      object_meta.cast_xywh()


# shared for event_meta oid2object_meta and obj_id_traj.pkl
class ObjectMeta(object):
  def __init__(self):
    self.start_frame = -1
    self.end_frame = -1 # inclusive

    # optional
    self.frame2bbx = {}
    self.format = 'x1y1x2y2'

  def parse(self, data):
    self.start_frame = data['start_frame']
    self.end_frame = data['end_frame']
    if 'trajectory' in data:
      self.frame2bbx = data['trajectory']

  def dump(self):
    ret_dict={
      'start_frame': self.start_frame,
      'end_frame': self.end_frame,
      'trajectory': self.frame2bbx,
      # 'object_type': self.obj,
      'format': self.format,
    }

    return ret_dict

  def cast_xywh(self):
    if self.format == 'xywh':
      return

    self.format = 'xywh'
    frame2bbx = {}
    for frame in self.frame2bbx:
      bbx = self.frame2bbx[frame]
      frame2bbx[frame] = [bbx[0], bbx[1], bbx[2]-bbx[0], bbx[3]-bbx[1]]
    self.frame2bbx = frame2bbx

  def cast_x1y1x2y2(self):
    if self.format == 'x1y1x2y2':
      return

    self.format = 'x1y1x2y2'
    frame2bbx = {}
    for frame in self.fram2bbx:
      bbx = self.frame2bbx[frame]
      frame2bbx[frame] = [bbx[0], bbx[1], bbx[0]+bbx[2], bbx[1]+bbxo[3]]
    self.frame2bbx = frame2bbx

  # in-place change!
  def clone_and_filter(self, start_frame, end_frame):
    object_meta = ObjectMeta()
    object_meta.start_frame = max(self.start_frame, start_frame)
    object_meta.end_frame = min(self.end_frame, end_frame)
    object_meta.format = 'x1y1x2y2'

    for frame in self.frame2bbx:
      if frame < start_frame or frame > end_frame:
        continue
      object_meta.frame2bbx[frame] = self.frame2bbx[frame]

    return object_meta


# obj_id_traj.pkl
class ObjIdTracklet(object):
  def __init__(self):
    self.oid2object_meta = {}

  def load(self, file):
    with open(file) as f:
      data = cPickle.load(f)
      for oid in data:
        meta = ObjectMeta()
        meta.parse(data[oid])
        self.oid2object_meta[oid] = meta

  def dump(self,file):
    ret_dict=dict()
    for oid,meta in self.oid2object_meta.iteritems():
      ret_dict[oid] = meta.dump()
    with open(file,"w") as f:
      cPickle.dump(ret_dict,f)


# frm_bbx.pkl
class FrmBbx(object):
  def __init__(self):
   self.frame2oid2bbx = {}
   self.format = 'x1y1x2y2'
   
  def load(self, file):
    with open(file) as f:
      self.frame2oid2bbx = cPickle.load(f)

  def dump(self,file):
    with open(file,"w") as f:
      cPickle.dump(self.frame2oid2bbx,f)


class PredictEventMeta(object):
  def __init__(self):
    self.event = ''
    self.start = None
    self.end = None
    self.score = -1.
    self.oid2object_meta = {}
    self.frame2bbx_conf = {}
    self.format = 'xywh'

  def parse(self, data):
    self.event = data['event_type']
    self.start = data['start_frame']
    self.end = data['end_frame']
    self.frame2bbx_conf = data['trajectory']
    objects = data['objects']
    for oid in objects:
      meta = ObjectMeta()
      meta.format = self.format
      meta.parse(objects[oid])
      self.oid2object_meta[oid] = meta
