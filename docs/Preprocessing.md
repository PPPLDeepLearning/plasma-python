# Raw data

The raw 0D data comes in a plain structured text format:

 1. Shot list: a 2 column CSV file having a unique identifier of a shot and a disruption time columns (-1 for non-disruptive). 
 1. Individual shot files: a 2 column CSV files having a time and a plasma current value columns. The time grid used is common, but the length os shots is rather different (for each file in the shot list)

See `plasma.jet_signals` for more details.

# Preprocessing

The goal of the preprocessing step is to go from the raw data to the higher level primitives: Shots, ShotLists. 
In addition, signal is cut, clipped and resampled (use univariate linear spline, log transformation on signal) and a `ttd` (time-to-disruption) variable is introduced. 

Certain shots are marked invalid depending on the magnitude of the plasma current.

Preprocessed results are saved in a numpy binary `npz` file.

The core methods are:
  1. `plasma.preprocessor.preprocess.get_signals_and_times_from_file`
  1. `plasma.preprocessor.preprocess.cut_and_resample_signals`
  1. `plasma.utils.processing.cut_and_resample_signal`


# Normalization

Shot normalization is done to address the problem of different scales of plasma signals which could potentially have a negative effect on the neural network training and inference.

Normalizers are trained on the training shots (requires one pass over data before the RNN training). Normalizer training essentially means extracting a set of statistics about shots and incorporating them into shot (mean, std, min-max).
Similarly to preprocessing step, an entire ShotList is split into sublists, a random sublist is picked, then stats are extracted on a shot-by-shot basis and saved in a normalizer object.

Example:

```python
class MeanVarNormalizer(Normalizer):
    def __init__(self,conf):
        Normalizer.__init__(self,conf)
        self.means = None
        self.stds = None
```  

Will contain lists of means and standard deviations of signals in the training shot list.

Normalization is implemented as a class hierarchy, wirth a base `plasma.preprocessor.Normalizer` class defining how stats are extracted and how training is perfromed. A set of specific normalization classes e.g. `MeanVarNormalizer`, `VarNormalizer` is derived from it, implementing different methods of shot normalization.
