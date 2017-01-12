from plasma.conf_parser import parameters
import os
import errno

try:
    conf = parameters('./conf.yaml')
except: 
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "conf.yaml")

#conf = parameters(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'conf.yaml'))
