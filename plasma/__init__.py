# import os
import logging
try:
    import absl.logging
    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

import warnings
# TODO(KGF): temporarily suppress numpy>=1.17.0 warning with TF<2.0.0
# ~6x tensorflow/python/framework/dtypes.py:529: FutureWarning ...
warnings.filterwarnings('ignore',
                        category=FutureWarning,
                        message=r"passing \(type, 1\) or '1type' as a synonym of type is deprecated",  # noqa
                        module="tensor*")

# Optional: disable the C-based library diagnostic info and warning messages:
# 2019-11-06 18:27:31.698908: I ...  dynamic library libcublas.so.10
# (independent from tf.logging.set_verbosity() diagnostic control)
# Must be set before first import of tensorflow v0.12+
# (either directly or via Keras backend)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Ref: https://github.com/tensorflow/tensorflow/issues/1258
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/38645250#38645250  # noqa
