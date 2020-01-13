from __future__ import print_function
from plasma.preprocessor.normalize import VarNormalizer as Normalizer
from plasma.primitives.shots import ShotList  # , Shot
from scipy import stats
import numpy as np
from pprint import pprint
import os
import matplotlib
matplotlib.use('Agg')  # for machines that don't have a display
import matplotlib.pyplot as plt  # noqa
from matplotlib import rc  # noqa
rc('font', **{'family': 'serif', 'sans-serif': ['Times']})
rc('text', usetex=True)


class PerformanceAnalyzer():
    def __init__(self, results_dir=None, shots_dir=None, i=0, T_min_warn=None,
                 T_max_warn=None, verbose=False, pred_ttd=False, conf=None):
        self.T_min_warn = T_min_warn
        self.T_max_warn = T_max_warn
        dt = conf['data']['dt']
        T_max_warn_def = int(round(conf['data']['T_warning']/dt))
        # int(round(conf['data']['T_min_warn']/dt))
        T_min_warn_def = conf['data']['T_min_warn']
        if T_min_warn is None:
            self.T_min_warn = T_min_warn_def
        if T_max_warn is None:
            self.T_max_warn = T_max_warn_def
        if self.T_max_warn < self.T_min_warn:
            # computation of statistics is only correct if T_max_warn is larger
            # than T_min_warn
            print("T max warn is too small: need to increase artificially.")
            self.T_max_warn = self.T_min_warn + 1
        self.verbose = verbose
        self.results_dir = results_dir
        self.shots_dir = shots_dir
        self.i = i
        self.pred_ttd = pred_ttd
        self.saved_conf = conf
        self.conf = conf

        self.pred_train = None
        self.truth_train = None
        self.disruptive_train = None
        self.shot_list_train = None

        self.pred_test = None
        self.truth_test = None
        self.disruptive_test = None
        self.shot_list_test = None

        self.p_thresh_range = None

        self.normalizer = None

    def get_metrics_vs_p_thresh(self, mode):
        if mode == 'train':
            all_preds = self.pred_train
            all_truths = self.truth_train
            all_disruptive = self.disruptive_train

        elif mode == 'test':
            all_preds = self.pred_test
            all_truths = self.truth_test
            all_disruptive = self.disruptive_test

        return self.get_metrics_vs_p_thresh_custom(
            all_preds, all_truths, all_disruptive)

    def get_metrics_vs_p_thresh_custom(self, all_preds, all_truths,
                                       all_disruptive):
        return self.get_metrics_vs_p_thresh_fast(
            all_preds, all_truths, all_disruptive)
        P_thresh_range = self.get_p_thresh_range()
        correct_range = np.zeros_like(P_thresh_range)
        accuracy_range = np.zeros_like(P_thresh_range)
        fp_range = np.zeros_like(P_thresh_range)
        missed_range = np.zeros_like(P_thresh_range)
        early_alarm_range = np.zeros_like(P_thresh_range)

        for i, P_thresh in enumerate(P_thresh_range):
            correct, accuracy, fp_rate, missed, early_alarm_rate = (
                self.summarize_shot_prediction_stats(
                    P_thresh, all_preds, all_truths, all_disruptive))
            correct_range[i] = correct
            accuracy_range[i] = accuracy
            fp_range[i] = fp_rate
            missed_range[i] = missed
            early_alarm_range[i] = early_alarm_rate

        return (correct_range, accuracy_range, fp_range, missed_range,
                early_alarm_range)

    def get_p_thresh_range(self):
        if np.any(self.p_thresh_range) is None:
            all_preds_tr = self.pred_train
            all_truths_tr = self.truth_train
            all_disruptive_tr = self.disruptive_train
            all_preds_te = self.pred_test
            all_truths_te = self.truth_test
            all_disruptive_te = self.disruptive_test

            early_th_tr, correct_th_tr, late_th_tr, nd_th_tr = (
                self.get_threshold_arrays(all_preds_tr, all_truths_tr,
                                          all_disruptive_tr))
            early_th_te, correct_th_te, late_th_te, nd_th_te = (
                self.get_threshold_arrays(all_preds_te, all_truths_te,
                                          all_disruptive_te))
            all_thresholds = np.sort(np.concatenate(
                (early_th_tr, correct_th_tr, late_th_tr, nd_th_tr, early_th_te,
                 correct_th_te, late_th_te, nd_th_te)))
            self.p_thresh_range = all_thresholds
        # print(np.unique(self.p_thresh_range))
        return self.p_thresh_range

    def get_metrics_vs_p_thresh_fast(self, all_preds, all_truths,
                                     all_disruptive):
        all_disruptive = np.array(all_disruptive)
        if self.pred_train is not None:
            p_thresh_range = self.get_p_thresh_range()
        else:
            early_th, correct_th, late_th, nd_th = self.get_threshold_arrays(
                all_preds, all_truths, all_disruptive)
            p_thresh_range = np.sort(np.concatenate(
                (early_th, correct_th, late_th, nd_th)))
        correct_range = np.zeros_like(p_thresh_range)
        accuracy_range = np.zeros_like(p_thresh_range)
        fp_range = np.zeros_like(p_thresh_range)
        missed_range = np.zeros_like(p_thresh_range)
        early_alarm_range = np.zeros_like(p_thresh_range)

        early_th, correct_th, late_th, nd_th = self.get_threshold_arrays(
            all_preds, all_truths, all_disruptive)

        for i, thresh in enumerate(p_thresh_range):
            correct, accuracy, fp_rate, missed, early_alarm_rate = (
                self.get_shot_prediction_stats_from_threshold_arrays(
                    early_th, correct_th, late_th, nd_th, thresh))
            correct_range[i] = correct
            accuracy_range[i] = accuracy
            fp_range[i] = fp_rate
            missed_range[i] = missed
            early_alarm_range[i] = early_alarm_rate

        return (correct_range, accuracy_range, fp_range, missed_range,
                early_alarm_range)

    def get_shot_prediction_stats_from_threshold_arrays(
            self, early_th, correct_th, late_th, nd_th, thresh):
        # indices = np.where(np.logical_and(
        #     correct_th > thresh, early_th <= thresh))[0]
        FPs = np.sum(nd_th > thresh)
        TNs = len(nd_th) - FPs

        earlies = np.sum(early_th > thresh)
        TPs = np.sum(np.logical_and(early_th <= thresh, correct_th > thresh))
        lates = np.sum(np.logical_and(np.logical_and(
            early_th <= thresh, correct_th <= thresh), late_th > thresh))
        FNs = np.sum(np.logical_and(np.logical_and(
            early_th <= thresh, correct_th <= thresh), late_th <= thresh))

        return self.get_accuracy_and_fp_rate_from_stats(
            TPs, FPs, FNs, TNs, earlies, lates)

    def get_shot_difficulty(self, preds, truths, disruptives):
        disruptives = np.array(disruptives)
        (d_early_thresholds, d_correct_thresholds, d_late_thresholds,
         nd_thresholds) = self.get_threshold_arrays(preds, truths, disruptives)
        d_thresholds = np.maximum(d_early_thresholds, d_correct_thresholds)
        # rank shots by difficulty. rank 1 is assigned to lowest value, should
        # be highest difficulty
        # difficulty is highest when threshold is low, can't detect disruption
        d_ranks = stats.rankdata(d_thresholds, method='min')
        # difficulty is highest when threshold is high, can't avoid false
        # positive
        nd_ranks = stats.rankdata(-nd_thresholds, method='min')
        ranking_fac = self.saved_conf['training']['ranking_difficulty_fac']
        facs_d = np.linspace(ranking_fac, 1, len(d_ranks))[d_ranks-1]
        facs_nd = np.linspace(ranking_fac, 1, len(nd_ranks))[nd_ranks-1]
        ret_facs = np.ones(len(disruptives))
        ret_facs[disruptives] = facs_d
        ret_facs[~disruptives] = facs_nd
        # print("setting shot difficulty")
        # print(disruptives)
        # print(d_thresholds)
        # print(nd_thresholds)
        # print(ret_facs)
        return ret_facs

    def get_threshold_arrays(self, preds, truths, disruptives):
        # num_d = np.sum(disruptives)
        # num_nd = np.sum(~disruptives)
        nd_thresholds = []
        d_early_thresholds = []
        d_correct_thresholds = []
        d_late_thresholds = []
        for i in range(len(preds)):
            pred = 1.0*preds[i]
            truth = truths[i]
            pred[:self.get_ignore_indices()] = -np.inf
            is_disruptive = disruptives[i]
            if is_disruptive:
                max_acceptable = self.create_acceptable_region(truth, 'max')
                min_acceptable = self.create_acceptable_region(truth, 'min')
                correct_indices = np.logical_and(
                    max_acceptable, ~min_acceptable)
                early_indices = ~max_acceptable
                late_indices = min_acceptable
                if np.sum(late_indices) == 0:
                    d_late_thresholds.append(-np.inf)
                else:
                    d_late_thresholds.append(np.max(pred[late_indices]))
                #
                if np.sum(early_indices) == 0:
                    d_early_thresholds.append(-np.inf)
                else:
                    d_early_thresholds.append(np.max(pred[early_indices]))
                #
                if np.sum(correct_indices) == 0:
                    d_correct_thresholds.append(-np.inf)
                else:
                    d_correct_thresholds.append(np.max(pred[correct_indices]))
            else:
                nd_thresholds.append(np.max(pred))
        return (np.array(d_early_thresholds), np.array(d_correct_thresholds),
                np.array(d_late_thresholds), np.array(nd_thresholds))

    def summarize_shot_prediction_stats_by_mode(self, P_thresh, mode,
                                                verbose=False):
        if mode == 'train':
            all_preds = self.pred_train
            all_truths = self.truth_train
            all_disruptive = self.disruptive_train

        elif mode == 'test':
            all_preds = self.pred_test
            all_truths = self.truth_test
            all_disruptive = self.disruptive_test

        return self.summarize_shot_prediction_stats(
            P_thresh, all_preds, all_truths, all_disruptive, verbose)

    def summarize_shot_prediction_stats(self, P_thresh, all_preds, all_truths,
                                        all_disruptive, verbose=False):
        TPs, FPs, FNs, TNs, earlies, lates = (0, 0, 0, 0, 0, 0)
        for i in range(len(all_preds)):
            preds = all_preds[i]
            truth = all_truths[i]
            is_disruptive = all_disruptive[i]
            TP, FP, FN, TN, early, late = self.get_shot_prediction_stats(
                P_thresh, preds, truth, is_disruptive)
            TPs += TP
            FPs += FP
            FNs += FN
            TNs += TN
            earlies += early
            lates += late

        disr = earlies + lates + TPs + FNs
        nondisr = FPs + TNs
        if verbose:
            print('total: {}, tp: {} fp: {} fn: {} tn: {} '.format(
                len(all_preds), TPs, FPs, FNs, TNs,),
                  'early: {} late: {} disr: {} nondisr: {}'.format(
                      earlies, lates, disr, nondisr))

        return self.get_accuracy_and_fp_rate_from_stats(
            TPs, FPs, FNs, TNs, earlies, lates, verbose)

    # we are interested in the predictions of the *first alarm*

    def get_shot_prediction_stats(self, P_thresh, pred, truth, is_disruptive):
        if self.pred_ttd:
            predictions = pred < P_thresh
        else:
            predictions = pred > P_thresh
        predictions = np.reshape(predictions, (len(predictions),))

        max_acceptable = self.create_acceptable_region(truth, 'max')
        min_acceptable = self.create_acceptable_region(truth, 'min')

        early = late = TP = TN = FN = FP = 0

        positives = self.get_positives(predictions)  # where(predictions)[0]
        if len(positives) == 0:
            if is_disruptive:
                FN = 1
            else:
                TN = 1
        else:
            if is_disruptive:
                first_pred_idx = positives[0]
                if (max_acceptable[first_pred_idx]
                        and ~min_acceptable[first_pred_idx]):
                    TP = 1
                elif min_acceptable[first_pred_idx]:
                    late = 1
                elif ~max_acceptable[first_pred_idx]:
                    early = 1
            else:
                FP = 1
        return TP, FP, FN, TN, early, late

    def get_ignore_indices(self):
        return self.saved_conf['model']['ignore_timesteps']

    def get_positives(self, predictions):
        indices = np.arange(len(predictions))
        return np.where(
            np.logical_and(
                predictions,
                indices >= self.get_ignore_indices()))[0]

    def create_acceptable_region(self, truth, mode):
        if mode == 'min':
            acceptable_timesteps = self.T_min_warn
        elif mode == 'max':
            acceptable_timesteps = self.T_max_warn
        else:
            print('Error Invalid Mode for acceptable region')
            exit(1)
        assert self.T_max_warn > self.T_min_warn

        acceptable = np.zeros_like(truth, dtype=bool)
        if acceptable_timesteps > 0:
            acceptable[-acceptable_timesteps:] = True
        return acceptable

    def get_accuracy_and_fp_rate_from_stats(self, tp, fp, fn, tn, early, late,
                                            verbose=False):
        total = tp + fp + fn + tn + early + late
        disr = early + late + tp + fn
        nondisr = fp + tn

        if disr == 0:
            early_alarm_rate = 0
            missed = 0
            accuracy = 0
        else:
            early_alarm_rate = 1.0*early/disr
            missed = 1.0*(late + fn)/disr
            accuracy = 1.0*tp/disr
        if nondisr == 0:
            fp_rate = 0
        else:
            fp_rate = 1.0*fp/nondisr
        correct = 1.0*(tp + tn)/total

        if verbose:
            print('accuracy: {}'.format(accuracy))
            print('missed: {}'.format(missed))
            print('early alarms: {}'.format(early_alarm_rate))
            print('false positive rate: {}'.format(fp_rate))
            print('correct: {}'.format(correct))

        return correct, accuracy, fp_rate, missed, early_alarm_rate

    def load_ith_file(self):
        results_files = os.listdir(self.results_dir)
        print(results_files)
        dat = np.load(self.results_dir + results_files[self.i],
                      allow_pickle=True)
        print("Loading results file {}".format(
            self.results_dir + results_files[self.i]))
        if self.verbose:
            print('configuration: {} '.format(dat['conf']))

        self.pred_train = dat['y_prime_train']
        self.truth_train = dat['y_gold_train']
        self.disruptive_train = dat['disruptive_train']
        self.pred_test = dat['y_prime_test']
        self.truth_test = dat['y_gold_test']
        self.disruptive_test = dat['disruptive_test']
        self.shot_list_test = ShotList(dat['shot_list_test'][()])
        self.shot_list_train = ShotList(dat['shot_list_train'][()])
        self.saved_conf = dat['conf'][()]
        # all files must agree on T_warning due to output of truth vs.
        # normalized shot ttd.
        self.conf['data']['T_warning'] = self.saved_conf['data']['T_warning']
        for mode in ['test', 'train']:
            print('{}: loaded {} shot ({}) disruptive'.format(
                mode, self.get_num_shots(mode),
                self.get_num_disruptive_shots(mode)))
        if self.verbose:
            self.print_conf()
        # self.assert_same_lists(self.shot_list_test, self.truth_test,
        # self.disruptive_test)
        # self.assert_same_lists(self.shot_list_train, self.truth_train,
        # self.disruptive_train)

    def assert_same_lists(self, shot_list, truth_arr, disr_arr):
        assert len(shot_list) == len(truth_arr)
        for i in range(len(shot_list)):
            # TODO(KGF): remove this hardcoded path (also missing signal_path))
            shot_list.shots[i].restore("/tigress/jk7/processed_shots/")
            s = shot_list.shots[i].ttd
            if not truth_arr[i].shape[0] == s.shape[0]-30:
                print(i)
                print(shot_list.shots[i].number)
                print((s.shape, truth_arr[i].shape, disr_arr[i]))
            assert truth_arr[i].shape[0] == s.shape[0]-30
        print("Same shape!")

    def print_conf(self):
        pprint(self.saved_conf)

    def get_num_shots(self, mode):
        if mode == 'test':
            return len(self.disruptive_test)
        if mode == 'train':
            return len(self.disruptive_train)

    def get_num_disruptive_shots(self, mode):
        if mode == 'test':
            return sum(self.disruptive_test)
        if mode == 'train':
            return sum(self.disruptive_train)

    def hist_alarms(self, alarms, title_str='alarms', save_figure=False,
                    linestyle='-'):
        fontsize = 15
        T_min_warn = self.T_min_warn
        T_max_warn = self.T_max_warn
        if len(alarms) > 0:
            alarms = alarms / 1000.0
            alarms = np.sort(alarms)
            T_min_warn /= 1000.0
            T_max_warn /= 1000.0
            plt.figure()
            alarms += 0.0001
            # bins = np.logspace(np.log10(min(alarms)), np.log10(max(alarms)),
            #                    40)

            # bins=linspace(min(alarms), max(alarms), 100)
            #        hist(alarms, bins=bins, alpha=1.0, histtype='step',
            #        normed=True, log=False, cumulative=-1)
            plt.step(np.concatenate((alarms[::-1], alarms[[0]])),
                     1.0*np.arange(alarms.size+1)/(alarms.size),
                     linestyle=linestyle, linewidth=1.5)
            plt.gca().set_xscale('log')
            plt.axvline(T_min_warn, color='r', linewidth=0.5)
            # if T_max_warn < np.max(alarms):
            #    plt.axvline(T_max_warn,color='r',linewidth=0.5)
            plt.xlabel('Time to disruption [s]', size=fontsize)
            plt.ylabel('Fraction of detected disruptions', size=fontsize)
            plt.xlim([1e-4, 4e1])  # max(alarms)*10])
            plt.ylim([0, 1])
            plt.grid()
            plt.title(title_str)
            plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize)
            plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize)
            plt.show()
            if save_figure:
                plt.savefig('accum_disruptions.png', dpi=200,
                            bbox_inches='tight')
        else:
            print(title_str + ": No alarms!")

    def gather_first_alarms(self, P_thresh, mode):
        if mode == 'train':
            pred_list = self.pred_train
            disruptive_list = self.disruptive_train
        elif mode == 'test':
            pred_list = self.pred_test
            disruptive_list = self.disruptive_test

        alarms = []
        disr_alarms = []
        nondisr_alarms = []
        for i in range(len(pred_list)):
            pred = pred_list[i]
            if self.pred_ttd:
                predictions = pred < P_thresh
            else:
                predictions = pred > P_thresh
            predictions = np.reshape(predictions, (len(predictions),))
            positives = self.get_positives(
                predictions)  # where(predictions)[0]
            if len(positives) > 0:
                alarm_ttd = len(pred) - 1.0 - positives[0]
                alarms.append(alarm_ttd)
                if disruptive_list[i]:
                    disr_alarms.append(alarm_ttd)
                else:
                    nondisr_alarms.append(alarm_ttd)
            else:
                if disruptive_list[i]:
                    disr_alarms.append(-1)
        return np.array(alarms), np.array(
            disr_alarms), np.array(nondisr_alarms)

    def compute_tradeoffs_and_print(self, mode):
        P_thresh_range = self.get_p_thresh_range()
        (correct_range, accuracy_range, fp_range, missed_range,
         early_alarm_range) = self.get_metrics_vs_p_thresh(mode)
        fp_threshs = [0.01, 0.05, 0.1]
        missed_threshs = [0.01, 0.05, 0.0]
        # missed_threshs = [0.01, 0.05, 0.1, 0.2, 0.3]

        # first index where...
        for fp_thresh in fp_threshs:
            print('============= FP RATE < {} ============='.format(fp_thresh))
            if(any(fp_range < fp_thresh)):
                idx = np.where(fp_range <= fp_thresh)[0][0]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(
                    P_thresh_opt, mode, verbose=True)
                print('============= AT P_THRESH = {} ============='.format(
                    P_thresh_opt))
            else:
                print('No such P_thresh found')
            print('')

        # last index where
        for missed_thresh in missed_threshs:
            print('============= MISSED RATE < {} ============='.format(
                missed_thresh))
            if(any(missed_range < missed_thresh)):
                idx = np.where(missed_range <= missed_thresh)[0][-1]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(
                    P_thresh_opt, mode, verbose=True)
                print('============= AT P_THRESH = {} ============='.format(
                    P_thresh_opt))
            else:
                print('No such P_thresh found')
            print('')

        print('============== Crossing Point: ==============')
        print('============= TEST PERFORMANCE: =============')
        idx = np.where(missed_range <= fp_range)[0][-1]
        P_thresh_opt = P_thresh_range[idx]
        self.summarize_shot_prediction_stats_by_mode(
            P_thresh_opt, mode, verbose=True)
        P_thresh_ret = P_thresh_opt
        return P_thresh_ret

    def compute_tradeoffs_and_print_from_training(self):
        P_thresh_range = self.get_p_thresh_range()
        (correct_range, accuracy_range, fp_range, missed_range,
         early_alarm_range) = self.get_metrics_vs_p_thresh('train')

        fp_threshs = [0.01, 0.05, 0.1]
        missed_threshs = [0.01, 0.05, 0.0]
#        missed_threshs = [0.01,0.05,0.1,0.2,0.3]
        P_thresh_default = 0.03
        P_thresh_ret = P_thresh_default

        first_idx = 0 if not self.pred_ttd else -1
        last_idx = -1 if not self.pred_ttd else 0

        # first index where...
        for fp_thresh in fp_threshs:
            print('============= TRAINING FP RATE < {} ============='.format(
                fp_thresh))
            print('============= TEST PERFORMANCE: =============')
            if(any(fp_range < fp_thresh)):
                idx = np.where(fp_range <= fp_thresh)[0][first_idx]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(
                    P_thresh_opt, 'test', verbose=True)
                print('============= AT P_THRESH = {} ============='.format(
                    P_thresh_opt))
            else:
                print('No such P_thresh found')
            P_thresh_opt = P_thresh_default
            print('')

        # last index where
        for missed_thresh in missed_threshs:
            print('============= TRAINING MISSED RATE < {} ==========='.format(
                missed_thresh))
            print('============= TEST PERFORMANCE: =============')
            if(any(missed_range < missed_thresh)):
                idx = np.where(missed_range <= missed_thresh)[0][last_idx]
                P_thresh_opt = P_thresh_range[idx]
                self.summarize_shot_prediction_stats_by_mode(
                    P_thresh_opt, 'test', verbose=True)
                if missed_thresh == 0.05:
                    P_thresh_ret = P_thresh_opt
                print('============= AT P_THRESH = {} ============='.format(
                    P_thresh_opt))
            else:
                print('No such P_thresh found')
            P_thresh_opt = P_thresh_default
            print('')

        print('============== Crossing Point: ==============')
        print('============= TEST PERFORMANCE: =============')
        if(any(missed_range <= fp_range)):
            idx = np.where(missed_range <= fp_range)[0][last_idx]
            P_thresh_opt = P_thresh_range[idx]
            self.summarize_shot_prediction_stats_by_mode(
                P_thresh_opt, 'test', verbose=True)
            P_thresh_ret = P_thresh_opt
            print('============= AT P_THRESH = {} ============='.format(
                P_thresh_opt))
        else:
            print('No such P_thresh found')
        return P_thresh_ret

    def compute_tradeoffs_and_plot(self, mode, save_figure=True,
                                   plot_string='', linestyle="-"):
        (correct_range, accuracy_range, fp_range, missed_range,
         early_alarm_range) = self.get_metrics_vs_p_thresh(mode)

        return self.tradeoff_plot(accuracy_range, missed_range, fp_range,
                                  early_alarm_range, save_figure=save_figure,
                                  plot_string=plot_string, linestyle=linestyle)

    def get_prediction_type(self, TP, FP, FN, TN, early, late):
        if TP:
            return 'TP'
        elif FP:
            return 'FP'
        elif FN:
            return 'FN'
        elif TN:
            return 'TN'
        elif early:
            return 'early'
        elif late:
            return 'late'

    def plot_individual_shot(self, P_thresh_opt, shot_num, normalize=True,
                             plot_signals=True):
        success = False
        for mode in ['test', 'train']:
            if mode == 'test':
                pred = self.pred_test
                truth = self.truth_test
                is_disruptive = self.disruptive_test
                shot_list = self.shot_list_test
            else:
                pred = self.pred_train
                truth = self.truth_train
                is_disruptive = self.disruptive_train
                shot_list = self.shot_list_train
            for i, shot in enumerate(shot_list):
                if shot.number == shot_num:
                    t = truth[i]
                    p = pred[i]
                    is_disr = is_disruptive[i]
                    TP, FP, FN, TN, early, late = (
                        self.get_shot_prediction_stats(P_thresh_opt, p, t,
                                                       is_disr))
                    prediction_type = self.get_prediction_type(TP, FP, FN, TN,
                                                               early, late)
                    print(prediction_type)
                    self.plot_shot(shot, True, normalize, t, p, P_thresh_opt,
                                   prediction_type, extra_filename='_indiv')
                    success = True
        if not success:
            print("Shot {} not found".format(shot_num))

    def get_prediction_type_for_individual_shot(self, P_thresh, shot,
                                                mode='test'):
        p, t, is_disr = self.get_pred_truth_disr_by_shot(shot)
        TP, FP, FN, TN, early, late = self.get_shot_prediction_stats(
            P_thresh, p, t, is_disr)
        prediction_type = self.get_prediction_type(TP, FP, FN, TN, early, late)
        return prediction_type

    def example_plots(self, P_thresh_opt, mode='test', types_to_plot=['FP'],
                      max_plot=5, normalize=True, plot_signals=True,
                      extra_filename=''):
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
            shot_list = self.shot_list_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
            shot_list = self.shot_list_train
        plotted = 0
        iterate_arr = np.arange(len(truth))
        np.random.shuffle(iterate_arr)
        for i in iterate_arr:
            t = truth[i]
            p = pred[i]
            is_disr = is_disruptive[i]
            shot = shot_list.shots[i]
            TP, FP, FN, TN, early, late = self.get_shot_prediction_stats(
                P_thresh_opt, p, t, is_disr)
            prediction_type = self.get_prediction_type(
                TP, FP, FN, TN, early, late)
            if not all(_ in set(['FP', 'TP', 'FN', 'TN', 'late',
                                 'early', 'any']) for _ in types_to_plot):
                print('warning, unkown types_to_plot')
                return
            if (('any' in types_to_plot or prediction_type in types_to_plot)
                    and plotted < max_plot):
                if plot_signals:
                    self.plot_shot(shot, True, normalize, t, p, P_thresh_opt,
                                   prediction_type,
                                   extra_filename=extra_filename)
                else:
                    plt.figure()
                    plt.semilogy((t+0.001)[::-1], label='ground truth')
                    plt.plot(p[::-1], 'g', label='neural net prediction')
                    plt.axvline(self.T_min_warn, color='r',
                                label='max warning time')
                    plt.axvline(self.T_max_warn, color='r',
                                label='min warning time')
                    plt.axhline(P_thresh_opt, color='k',
                                label='trigger threshold')
                    plt.xlabel('TTD [ms]')
                    plt.legend(loc=(1.0, 0.6))
                    plt.ylim([1e-7, 1.1e0])
                    plt.grid()
                    plt.savefig('fig_{}.png'.format(shot.number),
                                bbox_inches='tight')
                plotted += 1

    def plot_shot(self, shot, save_fig=True, normalize=True, truth=None,
                  prediction=None, P_thresh_opt=None, prediction_type='',
                  extra_filename=''):
        if self.normalizer is None and normalize:
            if self.conf is not None:
                self.saved_conf['paths']['normalizer_path'] = (
                    self.conf['paths']['normalizer_path'])
            nn = Normalizer(self.saved_conf)
            nn.train()
            self.normalizer = nn
            self.normalizer.set_inference_mode(True)

        if(shot.previously_saved(self.shots_dir)):
            shot.restore(self.shots_dir)
            if shot.signals_dict is not None:
                # make sure shot was saved with data
                # t_disrupt = shot.t_disrupt
                # is_disruptive = shot.is_disruptive
                if normalize:
                    self.normalizer.apply(shot)

                use_signals = self.saved_conf['paths']['use_signals']
                fontsize = 15
                lower_lim = 0  # len(pred)
                plt.close()
                # colors = ["b", "k"]
                # lss = ["-", "--"]
                f, axarr = plt.subplots(
                    len(use_signals)+1, 1, sharex=True, figsize=(10, 15))
                plt.title(prediction_type)
                assert np.all(shot.ttd.flatten() == truth.flatten())
                xx = range(len(prediction))  # list(reversed(range(len(pred))))
                for i, sig in enumerate(use_signals):
                    ax = axarr[i]
                    num_channels = sig.num_channels
                    sig_arr = shot.signals_dict[sig]
                    if num_channels == 1:
                        ax.plot(xx, sig_arr[:, 0], linewidth=2)
                        ax.plot([], linestyle="none", label=sig.description)
                        if np.min(sig_arr[:, 0]) < 0:
                            ax.set_ylim([-6, 6])
                            ax.set_yticks([-5, 0, 5])
                        ax.plot([], linestyle="none", label=sig.description)
                        if np.min(sig_arr[:, 0]) < 0:
                            ax.set_ylim([-6, 6])
                            ax.set_yticks([-5, 0, 5])
                        else:
                            ax.set_ylim([0, 8])
                            ax.set_yticks([0, 5])
                    else:
                        ax.imshow(sig_arr[:, :].T, aspect='auto',
                                  label=sig.description, cmap="inferno")
                        ax.set_ylim([0, num_channels])
                        ax.text(lower_lim+200, 45, sig.description,
                                bbox={'facecolor': 'white', 'pad': 10},
                                fontsize=fontsize-5)
                        ax.set_yticks([0, num_channels/2])
                        ax.set_yticklabels(["0", "0.5"])
                        ax.set_ylabel("$\\rho$", size=fontsize)
                    ax.legend(loc="best", labelspacing=0.1, fontsize=fontsize,
                              frameon=False)
                    ax.axvline(len(truth) - self.T_min_warn, color='r',
                               linewidth=0.5)
                    plt.setp(ax.get_xticklabels(), visible=False)
                    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                    f.subplots_adjust(hspace=0)
                ax = axarr[-1]
                # ax.semilogy((-truth+0.0001),label='ground truth')
                # ax.plot(-prediction+0.0001,'g',label='neural net prediction')
                # ax.axhline(-P_thresh_opt,color='k',label='trigger threshold')
                # nn = np.min(pred)
                ax.plot(xx, truth, 'g', label='target', linewidth=2)
                # ax.axhline(0.4,linestyle="--",color='k',label='threshold')
                ax.plot(xx, prediction, 'b', label='RNN output', linewidth=2)
                ax.axhline(P_thresh_opt, linestyle="--", color='k',
                           label='threshold')
                ax.set_ylim([-2, 2])
                ax.set_yticks([-1, 0, 1])
                # if len(truth)-T_max_warn >= 0:
                # ax.axvline(len(truth)-T_max_warn,color='r')#,label='max
                # warning time')
                # ,label='min warning time')
                ax.axvline(len(truth) - self.T_min_warn, color='r',
                           linewidth=0.5)
                ax.set_xlabel('T [ms]', size=fontsize)
                # ax.axvline(2400)
                ax.legend(loc=(0.5, 0.7), fontsize=fontsize-5,
                          labelspacing=0.1, frameon=False)
                plt.setp(ax.get_yticklabels(), fontsize=fontsize)
                plt.setp(ax.get_xticklabels(), fontsize=fontsize)
                # plt.xlim(0,200)
                plt.xlim([lower_lim, len(truth)])
        #         plt.savefig("{}.png".format(num),dpi=200,bbox_inches="tight")
                if save_fig:
                    plt.savefig('sig_fig_{}{}.png'.format(shot.number,
                                                          extra_filename),
                                bbox_inches='tight')
                    np.savez('sig_{}{}.npz'.format(shot.number,
                                                   extra_filename),
                             shot=shot, T_min_warn=self.T_min_warn,
                             T_max_warn=self.T_max_warn, prediction=prediction,
                             truth=truth, use_signals=use_signals,
                             P_thresh=P_thresh_opt)
                # plt.show()
        else:
            print("Shot hasn't been processed")

    def plot_shot_old(self, shot, save_fig=True, normalize=True, truth=None,
                      prediction=None, P_thresh_opt=None, prediction_type='',
                      extra_filename=''):
        if self.normalizer is None and normalize:
            if self.conf is not None:
                self.saved_conf['paths']['normalizer_path'] = (
                    self.conf['paths']['normalizer_path'])
            nn = Normalizer(self.saved_conf)
            nn.train()
            self.normalizer = nn
            self.normalizer.set_inference_mode(True)

        if shot.previously_saved(self.shots_dir):
            shot.restore(self.shots_dir)
            # t_disrupt = shot.t_disrupt
            # is_disruptive = shot.is_disruptive
            if normalize:
                self.normalizer.apply(shot)

            use_signals = self.saved_conf['paths']['use_signals']
            f, axarr = plt.subplots(len(use_signals)+1, 1, sharex=True,
                                    figsize=(13, 13))
            plt.title(prediction_type)
            # all files must agree on T_warning due to output of truth vs.
            # normalized shot ttd.
            assert np.all(shot.ttd.flatten() == truth.flatten())
            for i, sig in enumerate(use_signals):
                num_channels = sig.num_channels
                ax = axarr[i]
                sig_arr = shot.signals_dict[sig]
                if num_channels == 1:
                    ax.plot(sig_arr[:, 0], label=sig.description)
                else:
                    ax.imshow(sig_arr[:, :].T, aspect='auto',
                              label=sig.description + " (profile)")
                    ax.set_ylim([0, num_channels])
                ax.legend(loc='best', fontsize=8)
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), fontsize=7)
                f.subplots_adjust(hspace=0)
                # print(sig)
                # print('min: {}, max: {}'.format(np.min(sig_arr),
                # np.max(sig_arr)))
                ax = axarr[-1]
            if self.pred_ttd:
                ax.semilogy((-truth+0.0001), label='ground truth')
                ax.plot(-prediction+0.0001, 'g', label='neural net prediction')
                ax.axhline(-P_thresh_opt, color='k', label='trigger threshold')
            else:
                ax.plot((truth+0.001), label='ground truth')
                ax.plot(prediction, 'g', label='neural net prediction')
                ax.axhline(P_thresh_opt, color='k', label='trigger threshold')
            # ax.set_ylim([1e-5,1.1e0])
            ax.set_ylim([-2, 2])
            if len(truth)-self.T_max_warn >= 0:
                ax.axvline(len(truth)-self.T_max_warn, color='r',
                           label='min warning time')
            ax.axvline(len(truth) - self.T_min_warn, color='r',
                       label='max warning time')
            ax.set_xlabel('T [ms]')
            # ax.legend(loc = 'lower left',fontsize=10)
            plt.setp(ax.get_yticklabels(), fontsize=7)
            # ax.grid()
            if save_fig:
                plt.savefig('sig_fig_{}{}.png'.format(
                    shot.number, extra_filename), bbox_inches='tight')
                np.savez('sig_{}{}.npz'.format(shot.number, extra_filename),
                         shot=shot, T_min_warn=self.T_min_warn,
                         T_max_warn=self.T_max_warn, prediction=prediction,
                         truth=truth, use_signals=use_signals,
                         P_thresh=P_thresh_opt)
            plt.close()
        else:
            print("Shot hasn't been processed")

    def tradeoff_plot(self, accuracy_range, missed_range, fp_range,
                      early_alarm_range, save_figure=False, plot_string='',
                      linestyle="-"):
        fontsize = 15
        plt.figure()
        P_thresh_range = self.get_p_thresh_range()
        # semilogx(P_thresh_range,accuracy_range,label="accuracy")
        if self.pred_ttd:
            plt.semilogx(abs(P_thresh_range[::-1]), missed_range, 'r',
                         label="missed", linestyle=linestyle)
            plt.plot(abs(P_thresh_range[::-1]), fp_range, 'k',
                     label="false positives", linestyle=linestyle)
        else:
            plt.plot(P_thresh_range, missed_range, 'r', label="missed",
                     linestyle=linestyle)
            plt.plot(P_thresh_range, fp_range, 'k', label="false positives",
                     linestyle=linestyle)
        # plot(P_thresh_range,early_alarm_range,'c',label="early alarms")
        plt.legend(loc=(1.0, .6))
        plt.xlabel('Alarm threshold', size=fontsize)
        plt.grid()
        title_str = 'metrics{}'.format(plot_string.replace('_', ' '))
        plt.title(title_str)
        if save_figure:
            plt.savefig(title_str + '.png', bbox_inches='tight')
        plt.close('all')
        plt.plot(fp_range, 1-missed_range, '-b', linestyle=linestyle)
        ax = plt.gca()
        plt.xlabel('FP rate', size=fontsize)
        plt.ylabel('TP rate', size=fontsize)
        major_ticks = np.arange(0, 1.01, 0.2)
        minor_ticks = np.arange(0, 1.01, 0.05)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(minor_ticks, minor=True)
        plt.setp(plt.gca().get_yticklabels(), fontsize=fontsize)
        plt.setp(plt.gca().get_xticklabels(), fontsize=fontsize)
        ax.grid(which='both')
        ax.grid(which='major', alpha=0.5)
        ax.grid(which='minor', alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        if save_figure:
            plt.savefig(title_str + '_roc.png', bbox_inches='tight', dpi=200)
        print('ROC area ({}) is {}'.format(
            plot_string, self.roc_from_missed_fp(missed_range, fp_range)))
        return P_thresh_range, missed_range, fp_range

    def get_pred_truth_disr_by_shot(self, shot):
        if shot in self.shot_list_test:
            mode = 'test'
        elif shot in self.shot_list_train:
            mode = 'train'
        else:
            print('Shot {} not found'.format(shot))
            exit(1)
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
            shot_list = self.shot_list_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
            shot_list = self.shot_list_train
        i = shot_list.index(shot)
        t = truth[i]
        p = pred[i]
        is_disr = is_disruptive[i]
        shot = shot_list.shots[i]
        return p, t, is_disr

    def save_shot(self, shot, P_thresh_opt=0, extra_filename=''):
        if self.normalizer is None:
            if self.conf is not None:
                self.saved_conf['paths']['normalizer_path'] = (
                    self.conf['paths']['normalizer_path'])
            nn = Normalizer(self.saved_conf)
            nn.train()
            self.normalizer = nn
            self.normalizer.set_inference_mode(True)

        shot.restore(self.shots_dir)
        # t_disrupt = shot.t_disrupt
        # is_disruptive = shot.is_disruptive
        self.normalizer.apply(shot)

        pred, truth, is_disr = self.get_pred_truth_disr_by_shot(shot)
        use_signals = self.saved_conf['paths']['use_signals']
        np.savez('sig_{}{}.npz'.format(shot.number, extra_filename),
                 shot=shot, T_min_warn=self.T_min_warn,
                 T_max_warn=self.T_max_warn, prediction=pred,
                 truth=truth, use_signals=use_signals, P_thresh=P_thresh_opt)

    def get_roc_area_by_mode(self, mode='test'):
        if mode == 'test':
            pred = self.pred_test
            truth = self.truth_test
            is_disruptive = self.disruptive_test
            # shot_list = self.shot_list_test
        else:
            pred = self.pred_train
            truth = self.truth_train
            is_disruptive = self.disruptive_train
            # shot_list = self.shot_list_train
        return self.get_roc_area(pred, truth, is_disruptive)

    def get_roc_area(self, all_preds, all_truths, all_disruptive):
        (correct_range, accuracy_range, fp_range, missed_range,
         early_alarm_range) = self.get_metrics_vs_p_thresh_custom(
             all_preds, all_truths, all_disruptive)

        return self.roc_from_missed_fp(missed_range, fp_range)

    def roc_from_missed_fp(self, missed_range, fp_range):
        return -np.trapz(1 - missed_range, x=fp_range)
