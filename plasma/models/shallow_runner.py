from sklearn import svm
from keras.utils.generic_utils import Progbar
import keras.callbacks as cbks
from sklearn.metrics import classification_report
#  accuracy_score, auc, confusion_matrix
import joblib
from sklearn.ensemble import RandomForestClassifier
import hashlib
from plasma.utils.downloading import makedirs_process_safe
# from plasma.utils.state_reset import reset_states
from plasma.utils.evaluation import get_loss_from_list
from plasma.utils.performance import PerformanceAnalyzer
from plasma.utils.diagnostics import print_shot_list_sizes
# from plasma.models.loader import Loader, ProcessGenerator
# from plasma.conf import conf
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pathos.multiprocessing as mp
from functools import partial
import os
import datetime
import time
import numpy as np

from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import sys
# if sys.version_info[0] < 3:
#     from itertools import imap

# leading to import errors:
# from hyperopt import hp, STATUS_OK
# from hyperas.distributions import conditional

debug_use_shots = 100000
model_filename = "saved_model.pkl"
dataset_path = "dataset.npz"
dataset_test_path = "dataset_test.npz"


class FeatureExtractor(object):
    def __init__(self, loader, timesteps=32):
        self.loader = loader
        self.timesteps = timesteps
        self.positional_fit_order = 4
        self.num_positional_features = self.positional_fit_order + 1 + 3
        self.temporal_fit_order = 3
        self.num_temporal_features = self.temporal_fit_order + 1 + 3

    def get_sample_probs(self, shot_list, num_samples):
        print("Calculating number of timesteps")
        timesteps_total, timesteps_d, timesteps_nd = shot_list.num_timesteps(
            self.loader.conf['paths']['processed_prepath'])
        print("Total data: {} time samples, {} disruptive".format(
            timesteps_total, 1.0*timesteps_d/timesteps_total))
        if self.loader.conf['data']['equalize_classes']:
            sample_prob_d = np.minimum(1.0, 1.0*timesteps_nd/timesteps_d)
            sample_prob_nd = np.minimum(1.0, 1.0*timesteps_d/timesteps_nd)
            timesteps_total = (1.0*sample_prob_d*timesteps_d
                               + sample_prob_nd*timesteps_nd)
            sample_prob = np.minimum(1.0, 1.0*num_samples/timesteps_total)
            sample_prob_d *= sample_prob
            sample_prob_nd *= sample_prob
        else:
            sample_prob_d = np.minimum(1.0, 1.0*num_samples/timesteps_total)
            sample_prob_nd = sample_prob_d
        if sample_prob_nd <= 0.0 or sample_prob_d <= 0.0:
            val = np.minimum(1.0, num_samples/timesteps_total)
            return val, val
        return sample_prob_d, sample_prob_nd

    def load_shots(self, shot_list, is_inference=False, as_list=False,
                   num_samples=np.Inf):
        X = []
        Y = []
        Disr = []
        print("loading...")
        pbar = Progbar(len(shot_list))

        sample_prob_d, sample_prob_nd = self.get_sample_probs(
            shot_list, num_samples)
        fn = partial(
            self.load_shot,
            is_inference=is_inference,
            sample_prob_d=sample_prob_d,
            sample_prob_nd=sample_prob_nd)
        pool = mp.Pool()
        print('loading data in parallel on {} processes'.format(
            pool._processes))
        for x, y, disr in pool.imap(fn, shot_list):
            X.append(x)
            Y.append(y)
            Disr.append(disr)
            pbar.add(1.0)
        pool.close()
        pool.join()
        return X, Y, np.array(Disr)

    def get_save_prepath(self):
        prepath = self.loader.conf['paths']['processed_prepath']
        use_signals = self.loader.conf['paths']['use_signals']
        identifying_tuple = ''.join(
            tuple(map(lambda x: x.description,
                      sorted(use_signals)))).encode('utf-8')
        save_prepath = (
            prepath + "shallow/use_signals_{}/".format(
                int(hashlib.md5(identifying_tuple).hexdigest(), 16))
            )
        return save_prepath

    def process(self, shot):
        save_prepath = self.get_save_prepath()
        save_path = shot.get_save_path(save_prepath)
        if not os.path.exists(save_prepath):
            makedirs_process_safe(save_prepath)
        prepath = self.loader.conf['paths']['processed_prepath']
        assert shot.valid
        shot.restore(prepath)
        self.loader.set_inference_mode(True)  # make sure shots aren't cut
        if self.loader.normalizer is not None:
            self.loader.normalizer.apply(shot)
        else:
            print('Warning, no normalization. ',
                  'Training data may be poorly conditioned')
        self.loader.set_inference_mode(False)
        # sig, res = self.get_signal_result_from_shot(shot)
        disr = 1 if shot.is_disruptive else 0

        if not os.path.isfile(save_path):
            X = self.get_X(shot)
            np.savez(save_path, X=X)  # , Y=Y, disr=disr
            # print(X.shape, Y.shape)
        else:
            try:
                dat = np.load(save_path, allow_pickle=False)
                # X, Y, disr = dat["X"], dat["Y"], dat["disr"][()]
                X = dat["X"]
            except BaseException:
                # data was there but corrupted, save it again
                X = self.get_X(shot)
                np.savez(save_path, X=X)

        Y = self.get_Y(shot)

        shot.make_light()

        return X, Y, disr

    def get_X(self, shot):
        use_signals = self.loader.conf['paths']['use_signals']
        sig_sample = shot.signals_dict[use_signals[0]]
        if len(shot.ttd.shape) == 1:
            shot.ttd = np.expand_dims(shot.ttd, axis=1)
        length = sig_sample.shape[0]
        if length < self.timesteps:
            print(shot.ttd, shot.number)
            print("Shot must be at least as long as the RNN length.")
            exit(1)
        assert len(sig_sample.shape) == len(shot.ttd.shape) == 2
        assert shot.ttd.shape[1] == 1

        X = []
        while(len(X) == 0):
            for i in range(length-self.timesteps+1):
                # if np.random.rand() < sample_prob:
                x = self.get_x(i, shot)
                X.append(x)
        X = np.stack(X)
        return X

    def get_Y(self, shot):
        if len(shot.ttd.shape) == 1:
            shot.ttd = np.expand_dims(shot.ttd, axis=1)
        offset = self.timesteps - 1
        return np.round(shot.ttd[offset:, 0]).astype(np.int)

    def load_shot(self, shot, is_inference=False, sample_prob_d=1.0,
                  sample_prob_nd=1.0):
        X, Y, disr = self.process(shot)

        # cut shot ends if we are supposed to
        if self.loader.conf['data']['cut_shot_ends'] and not is_inference:
            T_min_warn = self.loader.conf['data']['T_min_warn']
            X = X[:-T_min_warn]
            Y = Y[:-T_min_warn]

        sample_prob = sample_prob_nd
        if disr:
            sample_prob = sample_prob_d
        if sample_prob < 1.0:
            indices = np.sort(np.random.choice(np.array(range(len(Y))),
                                               int(round(sample_prob*len(Y))),
                                               replace=False))
            X = X[indices]
            Y = Y[indices]
        return X, Y, disr

    def get_x(self, timestep, shot):
        x = []
        use_signals = self.loader.conf['paths']['use_signals']
        for sig in use_signals:
            x += [self.extract_features(timestep, shot, sig)]
            # x = sig[timestep:timestep + timesteps,:]
        x = np.concatenate(x, axis=0)
        return x

    # def get_x_y(self,timestep,shot):
    #     x = []
    #     use_signals = self.loader.conf['paths']['use_signals']
    #     for sig in use_signals:
    #         x += [self.extract_features(timestep,shot,sig)]
    #         # x = sig[timestep:timestep+timesteps,:]
    #     x = np.concatenate(x,axis=0)
    #     y = np.round(shot.ttd[timestep+self.timesteps-1,0]).astype(np.int)
    #     return x,y

    def extract_features(self, timestep, shot, signal):
        raw_sig = shot.signals_dict[signal][timestep:timestep + self.timesteps]
        num_positional_features = (
            self.num_positional_features if signal.num_channels > 1 else 1)
        output_arr = np.empty((self.timesteps, num_positional_features))
        final_output_arr = np.empty(
            (num_positional_features*self.num_temporal_features))
        for t in range(self.timesteps):
            output_arr[t, :] = self.extract_positional_features(raw_sig[t, :])
        for i in range(num_positional_features):
            idx = i*self.num_temporal_features
            final_output_arr[idx:idx + self.num_temporal_features] = (
                self.extract_temporal_features(output_arr[:, i]))
        return final_output_arr

    def extract_positional_features(self, arr):
        num_channels = len(arr)
        if num_channels > 1:
            ret_arr = np.empty(self.num_positional_features)
            coefficients = np.polynomial.polynomial.polyfit(
                np.linspace(0, 1, num_channels), arr,
                self.positional_fit_order)
            mu = np.mean(arr)
            std = np.std(arr)
            max_val = np.max(arr)
            ret_arr[:self.positional_fit_order+1] = coefficients
            ret_arr[self.positional_fit_order+1] = mu
            ret_arr[self.positional_fit_order+2] = std
            ret_arr[self.positional_fit_order+3] = max_val
            return ret_arr
        else:
            return arr

    def extract_temporal_features(self, arr):
        ret_arr = np.empty(self.num_temporal_features)
        coefficients = np.polynomial.polynomial.polyfit(
            np.linspace(0, 1, self.timesteps), arr, self.temporal_fit_order)
        mu = np.mean(arr)
        std = np.std(arr)
        max_val = np.max(arr)
        ret_arr[:self.temporal_fit_order+1] = coefficients
        ret_arr[self.temporal_fit_order+1] = mu
        ret_arr[self.temporal_fit_order+2] = std
        ret_arr[self.temporal_fit_order+3] = max_val
        return ret_arr

    def prepend_timesteps(self, arr):
        prepend = arr[0]*np.ones(self.timesteps-1)
        return np.concatenate((prepend, arr))


def build_callbacks(conf):
    '''
    The purpose of the method is to set up logging and history. It is based on
    Keras Callbacks
    https://github.com/fchollet/keras/blob/fbc9a18f0abc5784607cd4a2a3886558efa3f794/keras/callbacks.py

    Currently used callbacks include: BaseLogger, CSVLogger, EarlyStopping.
    Other possible callbacks to add in future:
    RemoteMonitor, LearningRateScheduler

    Argument list:
        - conf: There is a "callbacks" section in conf.yaml file.

    Relevant parameters are:
        list: Parameter specifying additional callbacks, read in the driver
    script and passed as an argument of type list (see next arg)
        metrics: List of quantities monitored during training and
    validation
        mode: one of {auto, min, max}. The decision to overwrite the
    current save file is made based on either the maximization or the
    minimization of the monitored quantity. For val_acc, this should be max,
    for val_loss this should be min, etc. In auto mode, the direction is
    automatically inferred from the name of the monitored quantity.
        monitor: Quantity used for early stopping, has to be from the list
    of metrics
        patience: Number of epochs used to decide on whether to apply early
    stopping or continue training

        - callbacks_list: uses callbacks.list configuration parameter,
          specifies the list of additional callbacks

    Returns:
        modified list of callbacks
    '''

    # mode = conf['callbacks']['mode']
    # monitor = conf['callbacks']['monitor']
    # patience = conf['callbacks']['patience']
    csvlog_save_path = conf['paths']['csvlog_save_path']
    # CSV callback is on by default
    if not os.path.exists(csvlog_save_path):
        os.makedirs(csvlog_save_path)

    # callbacks_list = conf['callbacks']['list']

    callbacks = [cbks.BaseLogger()]
    callbacks += [cbks.CSVLogger("{}callbacks-{}.log".format(
        csvlog_save_path,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    ]
    return cbks.CallbackList(callbacks)


def train(conf, shot_list_train, shot_list_validate, loader,
          shot_list_test=None):
    np.random.seed(1)
    print_shot_list_sizes(shot_list_train, shot_list_validate)
    print('training: {} shots, {} disruptive'.format(
        len(shot_list_train),
        shot_list_train.num_disruptive()))
    print('validate: {} shots, {} disruptive'.format(
        len(shot_list_validate),
        shot_list_validate.num_disruptive()))

    num_samples = conf['model']['shallow_model']['num_samples']
    feature_extractor = FeatureExtractor(loader)
    shot_list_train = shot_list_train.random_sublist(debug_use_shots)
    X, Y, _ = feature_extractor.load_shots(
        shot_list_train, num_samples=num_samples)
    Xv, Yv, _ = feature_extractor.load_shots(
        shot_list_validate, num_samples=num_samples)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    Xv = np.concatenate(Xv, axis=0)
    Yv = np.concatenate(Yv, axis=0)

    # max_samples = 100000
    # num_samples = min(max_samples, len(Y))
    # indices = np.random.choice(np.array(range(len(Y))), num_samples,
    #          replace=False)
    # X = X[indices]
    # Y = Y[indices]

    print("fitting on {} samples, {} positive".format(len(X), np.sum(Y > 0)))
    callbacks = build_callbacks(conf)
    callback_metrics = conf['callbacks']['metrics']
    callbacks.set_params({
        'metrics': callback_metrics,
        })
    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)

    # save_prepath = feature_extractor.get_save_prepath()
    model_path = (conf['paths']['model_save_path']
                  + model_filename)  # save_prepath + model_filename
    makedirs_process_safe(conf['paths']['model_save_path'])
    model_conf = conf['model']['shallow_model']
    if not model_conf['skip_train'] or not os.path.isfile(model_path):

        start_time = time.time()
        if model_conf["scale_pos_weight"] != 1:
            scale_pos_weight_dict = {
                np.min(Y): 1, np.max(Y): model_conf["scale_pos_weight"]}
        else:
            scale_pos_weight_dict = None
        if model_conf['type'] == "svm":
            model = svm.SVC(probability=True,
                            C=model_conf["C"],
                            kernel=model_conf["kernel"],
                            class_weight=scale_pos_weight_dict)
        elif model_conf['type'] == "random_forest":
            model = RandomForestClassifier(
                n_estimators=model_conf["n_estimators"],
                max_depth=model_conf["max_depth"],
                class_weight=scale_pos_weight_dict,
                n_jobs=-1)
        elif model_conf['type'] == "xgboost":
            max_depth = model_conf["max_depth"]
            if max_depth is None:
                max_depth = 0
            model = XGBClassifier(
                max_depth=max_depth,
                learning_rate=model_conf['learning_rate'],
                n_estimators=model_conf["n_estimators"],
                scale_pos_weight=model_conf["scale_pos_weight"])
        elif model_conf['type'] == 'mlp':
            hidden_layer_sizes = tuple(reversed(
                [model_conf['final_hidden_layer_size']*2**x
                 for x in range(model_conf['num_hidden_layers'])]))
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=model_conf['learning_rate_mlp'],
                alpha=model_conf['mlp_regularization'])
        else:
            print("Unkown model type, exiting.")
            exit(1)
        model.fit(X, Y)
        joblib.dump(model, model_path)
        print("Fit model in {} seconds".format(time.time()-start_time))
    else:
        model = joblib.load(model_path)
        print("model exists.")

    Y_pred = model.predict(X)
    print("Train")
    print(classification_report(Y, Y_pred))
    Y_predv = model.predict(Xv)
    print("Validate")
    print(classification_report(Yv, Y_predv))
    if ('monitor_test' in conf['callbacks'].keys()
            and conf['callbacks']['monitor_test']):
        times = conf['callbacks']['monitor_times']
        roc_areas, losses = make_predictions_and_evaluate_multiple_times(
            conf, shot_list_validate, loader, times)
        for roc, t in zip(roc_areas, times):
            print('val_roc_{} = {}'.format(t, roc))
        if shot_list_test is not None:
            roc_areas, losses = make_predictions_and_evaluate_multiple_times(
                conf, shot_list_test, loader, times)
            for roc, t in zip(roc_areas, times):
                print('test_roc_{} = {}'.format(t, roc))
    # print(confusion_matrix(Y,Y_pred))
    _, _, _, roc_area, loss = make_predictions_and_evaluate_gpu(
        conf, shot_list_validate, loader)
    # _, _, _, roc_area_train, loss_train = make_predictions_and_evaluate_gpu(
    #          conf, shot_list_train, loader)

    print('Validation Loss: {:.3e}'.format(loss))
    print('Validation ROC: {:.4f}'.format(roc_area))
    epoch_logs = {}
    epoch_logs['val_roc'] = roc_area
    epoch_logs['val_loss'] = loss
    # epoch_logs['train_roc'] = roc_area_train
    # epoch_logs['train_loss'] = loss_train
    callbacks.on_epoch_end(0, epoch_logs)

    print('...done')


def make_predictions(conf, shot_list, loader, custom_path=None):
    feature_extractor = FeatureExtractor(loader)
    # save_prepath = feature_extractor.get_save_prepath()
    if custom_path is None:
        model_path = conf['paths']['model_save_path'] + \
            model_filename  # save_prepath + model_filename
    else:
        model_path = custom_path
    model = joblib.load(model_path)
    # shot_list = shot_list.random_sublist(10)

    y_prime = []
    y_gold = []
    disruptive = []

    pbar = Progbar(len(shot_list))
    fn = partial(
        predict_single_shot,
        model=model,
        feature_extractor=feature_extractor)
    pool = mp.Pool()
    print('predicting in parallel on {} processes'.format(pool._processes))
    # for (y_p, y, disr) in map(fn, shot_list):
    for (y_p, y, disr) in pool.imap(fn, shot_list):
        # y_p, y, disr = predict_single_shot(model, feature_extractor,shot)
        y_prime += [np.expand_dims(y_p, axis=1)]
        y_gold += [np.expand_dims(y, axis=1)]
        disruptive += [disr]
        pbar.add(1.0)

    pool.close()
    pool.join()
    return y_prime, y_gold, disruptive


def predict_single_shot(shot, model, feature_extractor):
    X, y, disr = feature_extractor.load_shot(shot, is_inference=True)
    y_p = model.predict_proba(X)[:, 1]
    # print(y)
    # print(y_p)
    y = feature_extractor.prepend_timesteps(y)
    y_p = feature_extractor.prepend_timesteps(y_p)
    return y_p, y, disr


def make_predictions_and_evaluate_gpu(
        conf, shot_list, loader, custom_path=None):
    y_prime, y_gold, disruptive = make_predictions(
        conf, shot_list, loader, custom_path)
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
    loss = get_loss_from_list(y_prime, y_gold, conf['data']['target'])
    return y_prime, y_gold, disruptive, roc_area, loss


def make_predictions_and_evaluate_multiple_times(conf, shot_list, loader,
                                                 times, custom_path=None):
    y_prime, y_gold, disruptive = make_predictions(conf, shot_list, loader,
                                                   custom_path)
    areas = []
    losses = []
    for T_min_curr in times:
        # if 'monitor_test' in conf['callbacks'].keys() and
        # conf['callbacks']['monitor_test']:
        conf_curr = deepcopy(conf)
        T_min_warn_orig = conf['data']['T_min_warn']
        conf_curr['data']['T_min_warn'] = T_min_curr
        assert conf['data']['T_min_warn'] == T_min_warn_orig
        analyzer = PerformanceAnalyzer(conf=conf_curr)
        roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
        # shot_list.set_weights(analyzer.get_shot_difficulty(y_prime, y_gold,
        # disruptive))
        loss = get_loss_from_list(y_prime, y_gold, conf['data']['target'])
        areas.append(roc_area)
        losses.append(loss)
    return areas, losses
