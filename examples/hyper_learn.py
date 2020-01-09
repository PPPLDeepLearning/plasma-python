# from plasma.models import runner
# from plasma.models.loader import Loader

# import numpy as np
# from hyperopt import Trials, tpe

# from plasma.conf import conf
# from pprint import pprint
# pprint(conf)
#  #from plasma.primitives.shots import Shot, ShotList
#  #from plasma.models.runner import train, make_predictions
#  ,make_predictions_gpu

# if conf['data']['normalizer'] == 'minmax':
#     from plasma.preprocessor.normalize import MinMaxNormalizer as Normalizer
# elif conf['data']['normalizer'] == 'meanvar':
#     from plasma.preprocessor.normalize import MeanVarNormalizer as Normalizer
# elif conf['data']['normalizer'] == 'var':
#     # performs !much better than minmaxnormalizer
#     from plasma.preprocessor.normalize import VarNormalizer as Normalizer
# elif conf['data']['normalizer'] == 'averagevar':
#     # performs !much better than minmaxnormalizer
#     from plasma.preprocessor.normalize import (
#         AveragingVarNormalizer as Normalizer
#     )
# else:
#     print('unkown normalizer. exiting')
#     exit(1)

# np.random.seed(1)

# print("normalization", end='')
# nn = Normalizer(conf)
# nn.train()
# loader = Loader(conf, nn)
# shot_list_train, shot_list_validate, shot_list_test = loader.load_shotlists(
#     conf)
# print("...done")

# print('Training on {} shots, testing on {} shots'.format(
#     len(shot_list_train), len(shot_list_test)))

# specific_runner = runner.HyperRunner(conf, loader, shot_list_train)

# best_run, best_model = specific_runner.frnn_minimize(
#     algo=tpe.suggest, max_evals=2, trials=Trials())
# print(best_run)
# print(best_model)
