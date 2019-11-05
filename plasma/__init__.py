import warnings
# TODO(KGF): temporarily suppress numpy>=1.17.0 warning with TF<2.0.0
# ~6x tensorflow/python/framework/dtypes.py:529: FutureWarning ...
warnings.filterwarnings('ignore',
                        category=FutureWarning,
                        message=r"passing \(type, 1\) or '1type' as a synonym of type is deprecated",  # noqa
                        module="tensorflow")
