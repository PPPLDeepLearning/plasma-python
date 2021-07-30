import collections

class CallbackList(object):
  """Container abstracting a list of callbacks.
  Arguments:
      callbacks: List of `Callback` instances.
      queue_length: Queue length for keeping
          running statistics over callback execution time.
  """

  def __init__(self, callbacks=None, queue_length=10):
    callbacks = callbacks or []
    self.callbacks = [c for c in callbacks]
    self.queue_length = queue_length
    self.params = {}
    self.model = None
    self._reset_batch_timing()

  def _reset_batch_timing(self):
    self._delta_t_batch = 0.
    self._delta_ts = collections.defaultdict(
        lambda: collections.deque([], maxlen=self.queue_length))

  def append(self, callback):
    self.callbacks.append(callback)

  def set_params(self, params):
    self.params = params
    for callback in self.callbacks:
      callback.set_params(params)

  def set_model(self, model):
    self.model = model
    for callback in self.callbacks:
      callback.set_model(model)

  def _call_batch_hook(self, mode, hook, batch, logs=None):
    """Helper function for all batch_{begin | end} methods."""
    if not self.callbacks:
      return
    hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)
    if hook == 'begin':
      self._t_enter_batch = time.time()
    if hook == 'end':
      # Batch is ending, calculate batch time.
      self._delta_t_batch = time.time() - self._t_enter_batch

    logs = logs or {}
    t_before_callbacks = time.time()
    for callback in self.callbacks:
      batch_hook = getattr(callback, hook_name)
      batch_hook(batch, logs)
    self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

    delta_t_median = np.median(self._delta_ts[hook_name])
    if (self._delta_t_batch > 0. and
        delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
      logging.warning(
          'Method (%s) is slow compared '
          'to the batch update (%f). Check your callbacks.', hook_name,
          delta_t_median)

  def _call_begin_hook(self, mode):
    """Helper function for on_{train|test|predict}_begin methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_begin()
    elif mode == ModeKeys.TEST:
      self.on_test_begin()
    else:
      self.on_predict_begin()

  def _call_end_hook(self, mode):
    """Helper function for on_{train|test|predict}_end methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_end()
    elif mode == ModeKeys.TEST:
      self.on_test_end()
    else:
      self.on_predict_end()

  def on_batch_begin(self, batch, logs=None):
    self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_batch_end(self, batch, logs=None):
    self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    """Calls the `on_epoch_begin` methods of its callbacks.
    This function should only be called during TRAIN mode.
    Arguments:
        epoch: integer, index of epoch.
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = logs or {}
    for callback in self.callbacks:
      callback.on_epoch_begin(epoch, logs)
    self._reset_batch_timing()

  def on_epoch_end(self, epoch, logs=None):
    """Calls the `on_epoch_end` methods of its callbacks.
    This function should only be called during TRAIN mode.
    Arguments:
        epoch: integer, index of epoch.
        logs: dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
    logs = logs or {}
    for callback in self.callbacks:
      callback.on_epoch_end(epoch, logs)

  def on_train_batch_begin(self, batch, logs=None):
    """Calls the `on_train_batch_begin` methods of its callbacks.
    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_train_batch_end(self, batch, logs=None):
    """Calls the `on_train_batch_end` methods of its callbacks.
    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Metric results for this batch.
    """
    self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_test_batch_begin(self, batch, logs=None):
    """Calls the `on_test_batch_begin` methods of its callbacks.
    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

  def on_test_batch_end(self, batch, logs=None):
    """Calls the `on_test_batch_end` methods of its callbacks.
    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Metric results for this batch.
    """
    self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

  def on_predict_batch_begin(self, batch, logs=None):
    """Calls the `on_predict_batch_begin` methods of its callbacks.
    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    """
    self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

  def on_predict_batch_end(self, batch, logs=None):
    """Calls the `on_predict_batch_end` methods of its callbacks.
    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Metric results for this batch.
    """
    self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

  def on_train_begin(self, logs=None):
    """Calls the `on_train_begin` methods of its callbacks.
    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    for callback in self.callbacks:
      callback.on_train_begin(logs)

  def on_train_end(self, logs=None):
    """Calls the `on_train_end` methods of its callbacks.
    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    for callback in self.callbacks:
      callback.on_train_end(logs)

  def on_test_begin(self, logs=None):
    """Calls the `on_test_begin` methods of its callbacks.
    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    for callback in self.callbacks:
      callback.on_test_begin(logs)

  def on_test_end(self, logs=None):
    """Calls the `on_test_end` methods of its callbacks.
    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    for callback in self.callbacks:
      callback.on_test_end(logs)

  def on_predict_begin(self, logs=None):
    """Calls the 'on_predict_begin` methods of its callbacks.
    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    for callback in self.callbacks:
      callback.on_predict_begin(logs)

  def on_predict_end(self, logs=None):
    """Calls the `on_predict_end` methods of its callbacks.
    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    for callback in self.callbacks:
      callback.on_predict_end(logs)

  def __iter__(self):
    return iter(self.callbacks)