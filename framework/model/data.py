class Reader(object):
  def yield_trn_batch(self, batch_size, **kwargs):
    raise NotImplementedError("""please customize yield_trn_batch""")

  def yield_val_batch(self, batch_size, **kwargs):
    raise NotImplementedError("""please customize yield_val_batch""")

  def yield_tst_batch(self, batch_size, **kwargs):
    raise NotImplementedError("""please customize yield_tst_batch""")

  def reset(self):
    pass
