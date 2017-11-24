import os,sys
import numpy as np

from plasma.utils.performance import *
from plasma.conf import conf

#mode = 'test'
file_num = 0
save_figure = True
pred_ttd = False

T_min_warn = 30 #None #take value from conf #30

verbose=False
if len(sys.argv) == 3:
    results_dir = sys.argv[1]
    num = int(sys.argv[2])
else:
    results_dir = conf['paths']['results_prepath']
    num = int(sys.argv[1])

print("loading results from {}".format(results_dir))
print("Plotting shot {}".format(num))
shots_dir = conf['paths']['processed_prepath']

analyzer = PerformanceAnalyzer(conf=conf,results_dir=results_dir,shots_dir=shots_dir,i = file_num,
T_min_warn = T_min_warn, verbose = verbose, pred_ttd=pred_ttd) 

analyzer.load_ith_file()
P_thresh_opt = analyzer.compute_tradeoffs_and_print_from_training()
analyzer.plot_individual_shot(P_thresh_opt,num)

