from plasma.conf_parser import parameters
from plasma.primitives.shots import ShotListFiles
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
conf['paths']['all_signals'] = all_signals#jet_signals#d3d_signals#all_signals
#make sure all 1D signals appear last!
conf['paths']['use_signals'] = fully_defined_signals#d3d_signals#fully_defined_signals #d3d_signals#fully_defined_signals# [ip,lm,li,dens,q95,energy,pin,pradcore]#,edens_profile,etemp_profile]#jet_signals#

#shot lists
#shot_list_dir = conf['paths']['shot_list_dir']
#shot_list_dir = '/cscratch/share/frnn/shot_lists/'
shot_list_dir = '/tigress/jk7/shot_lists/'

jet_carbon_wall = ShotListFiles(jet,shot_list_dir,['CWall_clear.txt','CFC_unint.txt'],'jet carbon wall data')
jet_iterlike_wall = ShotListFiles(jet,shot_list_dir,['ILW_unint.txt','BeWall_clear.txt'],'jet iter like wall data')
jet_full = ShotListFiles(jet,shot_list_dir,['ILW_unint.txt','BeWall_clear.txt','CWall_clear.txt','CFC_unint.txt'],'jet full data')

d3d_10000 = ShotListFiles(d3d,shot_list_dir,['d3d_clear_10000.txt','d3d_disrupt_10000.txt'],'d3d data 10000 ND and D shots')
d3d_1000 = ShotListFiles(d3d,shot_list_dir,['d3d_clear_1000.txt','d3d_disrupt_1000.txt'],'d3d data 1000 ND and D shots')
d3d_100 = ShotListFiles(d3d,shot_list_dir,['d3d_clear_100.txt','d3d_disrupt_100.txt'],'d3d data 100 ND and D shots')
d3d_full = ShotListFiles(d3d,shot_list_dir,['d3d_clear_data_avail.txt','d3d_disrupt_data_avail.txt'],'d3d data since shot 125500')
d3d_jb_full = ShotListFiles(d3d,shot_list_dir,['shotlist_JaysonBarr_clear.txt','shotlist_JaysonBarr_disrupt.txt'],'d3d shots since 160000-170000')

nstx_full = ShotListFiles(nstx,shot_list_dir,['disrupt_nstx.txt'],'nstx shots (all are disruptive')

conf['paths']['shot_files'] = [jet_carbon_wall]#d3d_full,jet_iterlike_wall,jet_carbon_wall]#,jet_iterlike_wall,jet_carbon_wall]#,jet_iterlike_wall]#[d3d_full]#[jet_carbon_wall]
conf['paths']['shot_files_test'] = [jet_iterlike_wall]#[jet_iterlike_wall]#[d3d_full]#[jet_iterlike_wall]
conf['paths']['shot_files_all'] = conf['paths']['shot_files']+conf['paths']['shot_files_test']
conf['paths']['all_machines'] = list(set([file.machine for file in conf['paths']['shot_files_all']]))

#shot_numbers_files = ['d3d_short_clear.txt']# ,'d3d_clear.txt', 'd3d_disrupt.txt']
#shot_numbers_files = ['d3d_clear.txt', 'd3d_disrupt.txt']#['d3d_short_clear.txt']# ]


