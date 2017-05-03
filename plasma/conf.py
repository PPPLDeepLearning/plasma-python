from plasma.conf_parser import parameters
import os
import errno

if os.path.exists(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../examples/conf.yaml')):
    conf = parameters(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../examples/conf.yaml'))
elif os.path.exists('./conf.yaml'):
    conf = parameters('./conf.yaml')
elif os.path.exists('./examples/conf.yaml'):
    conf = parameters('./examples/conf.yaml')
elif os.path.exists('../examples/conf.yaml'):
    conf = parameters('../examples/conf.yaml')    
else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), 'conf.yaml')

from data.signals import *#d3d,jet,d3d_signals,jet_signals,all_signals
conf['paths']['all_signals'] = d3d_signals
conf['paths']['use_signals'] = [ip,lm,etemp_profile]
conf['paths']['all_machines'] = all_machines
conf['paths']['shot_files'] = [(jet,'CWall_clear.txt'),(jet,'CFC_unint.txt')]
conf['paths']['shot_files_test'] =  [(jet,'BeWall_clear.txt'),(jet,'ILW_unint.txt')]