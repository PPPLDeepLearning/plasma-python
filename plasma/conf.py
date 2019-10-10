from plasma.conf_parser import parameters
import os
import errno

# TODO(KGF): this conf.py feels like an unnecessary level of indirection
if os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               '../examples/conf.yaml')):
    conf = parameters(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                   '../examples/conf.yaml'))
elif os.path.exists('./conf.yaml'):
    conf = parameters('./conf.yaml')
elif os.path.exists('./examples/conf.yaml'):
    conf = parameters('./examples/conf.yaml')
elif os.path.exists('../examples/conf.yaml'):
    conf = parameters('../examples/conf.yaml')
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                            'conf.yaml')
