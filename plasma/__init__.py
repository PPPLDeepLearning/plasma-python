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
                        module="tensorflow")
