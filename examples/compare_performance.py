import sys
from plasma.utils.performance import PerformanceAnalyzer
from plasma.conf import conf

# mode = 'test'
file_num = 0
save_figure = True
pred_ttd = False

# cut_shot_ends = conf['data']['cut_shot_ends']
# dt = conf['data']['dt']
# T_max_warn = int(round(conf['data']['T_warning']/dt))
# T_min_warn = conf['data']['T_min_warn']
# T_min_warn = int(round(conf['data']['T_min_warn']/dt))
# if cut_shot_ends:
# 	T_max_warn = T_max_warn-T_min_warn
# 	T_min_warn = 0
T_min_warn = 30  # None #take value from conf #30

verbose = False
assert(sys.argv > 1)
results_dirs = sys.argv[1:]
shots_dir = conf['paths']['processed_prepath']

analyzers = [PerformanceAnalyzer(conf=conf, results_dir=results_dir,
                                 shots_dir=shots_dir, i=file_num,
                                 T_min_warn=T_min_warn,
                                 verbose=verbose, pred_ttd=pred_ttd)
             for results_dir in results_dirs]

for analyzer in analyzers:
    analyzer.load_ith_file()
    analyzer.verbose = False

P_threshs = [analyzer.compute_tradeoffs_and_print_from_training()
             for analyzer in analyzers]

print('Test ROC:')
for analyzer in analyzers:
    print(analyzer.get_roc_area_by_mode('test'))
# P_thresh_opt = 0.566#0.566#0.92#
# analyzer.compute_tradeoffs_and_print_from_training()
linestyle = "-"

# analyzer.compute_tradeoffs_and_plot('test', save_figure=save_figure,
#     plot_string='_test',linestyle=linestyle)
# analyzer.compute_tradeoffs_and_plot('train', save_figure=save_figure,
#     plot_string='_train',linestyle=linestyle)
# analyzer.summarize_shot_prediction_stats_by_mode(P_thresh_opt,'test')
shots = analyzers[0].shot_list_test

for shot in shots:
    if all([(shot in analyzer.shot_list_test
             or shot in analyzer.shot_list_train)
            for analyzer in analyzers]):
        types = [
            analyzers[i].get_prediction_type_for_individual_shot(
                P_threshs[i], shot, mode='test')
            for i in range(len(analyzers))]
        if types == ['TP', 'late']:
            if shot in analyzers[1].shot_list_test:
                print("TEST")
            else:
                print("TRAIN")
            print(shot.number)
            print(types)
            for i, analyzer in enumerate(analyzers):
                analyzer.save_shot(shot, P_thresh_opt=P_threshs[i],
                                   extra_filename=['1D', '0D'][i])
    else:
        pass
        # print("shot {} not in train or test shot list
        #                 (must be in validation)".format(shot))
