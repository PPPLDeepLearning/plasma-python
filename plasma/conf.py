from plasma.conf_parser import parameters
from plasma.primitives.shots import ShotListFile
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
#signals
conf['paths']['all_signals'] = d3d_signals
conf['paths']['use_signals'] = [ip,lm,etemp_profile]

#machines
conf['paths']['all_machines'] = all_machines

#shot lists
shot_list_dir = conf['paths']['shot_list_dir']

jet_carbon_wall = ShotListFile(jet,shot_list_dir,['CWall_clear.txt','CFC_unint.txt'],'jet carbon wall data')
jet_iterlike_wall = ShotListFile(jet,shot_list_dir,['ILW_unint.txt','BeWall_clear.txt'],'jet iter like wall data')
jet_full = ShotListFile(jet,shot_list_dir,['ILW_unint.txt','BeWall_clear.txt','CWall_clear.txt','CFC_unint.txt'],'jet full data')

d3d_full = ShotListFile(d3d,shot_list_dir,['d3d_clear_data_avail.txt','d3d_disrupt_data_avail.txt'],'d3d data since shot 125500')
d3d_jb_full = ShotListFile(d3d,shot_list_dir,['shotlist_JaysonBarr_clear.txt','shotlist_JaysonBarr_disrupt.txt'],'d3d shots since 160000-170000')

nstx_full = ShotListFile(nstx,shot_list_dir,['disrupt_nstx.txt'],'nstx shots (all are disruptive')

conf['paths']['shot_files'] = [d3d_full]#[jet_carbon_wall]
conf['paths']['shot_files_test'] = []#[jet_iterlike_wall]
conf['paths']['shot_files_all'] = conf['paths']['shot_files']+conf['paths']['shot_files_test']

#shot_numbers_files = ['d3d_short_clear.txt']# ,'d3d_clear.txt', 'd3d_disrupt.txt']
#shot_numbers_files = ['d3d_clear.txt', 'd3d_disrupt.txt']#['d3d_short_clear.txt']# ]


