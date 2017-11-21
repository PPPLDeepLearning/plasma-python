import os,sys
import numpy as np

from plasma.utils.performance import *
from plasma.conf import conf

#mode = 'test'
file_num = 0
save_figure = True
pred_ttd = False

# cut_shot_ends = conf['data']['cut_shot_ends']
# dt = conf['data']['dt']
# T_max_warn = int(round(conf['data']['T_warning']/dt))
# T_min_warn = conf['data']['T_min_warn']#int(round(conf['data']['T_min_warn']/dt))
# if cut_shot_ends:
# 	T_max_warn = T_max_warn-T_min_warn
# 	T_min_warn = 0
T_min_warn = 10 #None #take value from conf #30

verbose=False
if len(sys.argv) > 1:
    results_dir = sys.argv[1]
else:
    results_dir = conf['paths']['results_prepath']
shots_dir = conf['paths']['processed_prepath']

analyzer = PerformanceAnalyzer(conf=conf,results_dir=results_dir,shots_dir=shots_dir,i = file_num,
T_min_warn = T_min_warn, verbose = verbose, pred_ttd=pred_ttd) 

analyzer.load_ith_file()

P_thresh_opt = analyzer.compute_tradeoffs_and_print_from_training()

analyzer.compute_tradeoffs_and_plot('test',save_figure=save_figure,plot_string='_test')
analyzer.compute_tradeoffs_and_plot('train',save_figure=save_figure,plot_string='_train')

analyzer.summarize_shot_prediction_stats_by_mode(P_thresh_opt,'test')

#analyzer.example_plots(P_thresh_opt,'test','any')
analyzer.example_plots(P_thresh_opt,'test',['FP'],extra_filename='test')
analyzer.example_plots(P_thresh_opt,'test',['FN'],extra_filename='test')
analyzer.example_plots(P_thresh_opt,'test',['TP'],extra_filename='test')
analyzer.example_plots(P_thresh_opt,'test',['late'],extra_filename='test')

analyzer.example_plots(P_thresh_opt,'train',['TN'],extra_filename='train')
analyzer.example_plots(P_thresh_opt,'train',['FP'],extra_filename='train')
analyzer.example_plots(P_thresh_opt,'train',['FN'],extra_filename='train')
analyzer.example_plots(P_thresh_opt,'train',['TP'],extra_filename='train')
analyzer.example_plots(P_thresh_opt,'train',['late'],extra_filename='train')


alarms,disr_alarms,nondisr_alarms = analyzer.gather_first_alarms(P_thresh_opt,'test')
analyzer.hist_alarms(disr_alarms,'disruptive alarms, P thresh = {}'.format(P_thresh_opt),save_figure=save_figure)
print('{} disruptive alarms'.format(len(disr_alarms)))
print('{} seconds mean alarm time'.format(np.mean(disr_alarms[disr_alarms > 0])))
print('{} seconds median alarm time'.format(np.median(disr_alarms[disr_alarms > 0])))
analyzer.hist_alarms(nondisr_alarms,'nondisruptive alarms, P thresh = {}'.format(P_thresh_opt))
print('{} nondisruptive alarms'.format(len(nondisr_alarms)))

