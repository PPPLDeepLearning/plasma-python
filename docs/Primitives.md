## Shot

Each shot is a measurement of plasma current as a function of time. The Shot objects contains following attributes:

 1. number - integer, unique identifier of a shot
 1. t_disrupt - double, disruption time in milliseconds (second column in the shotlist input file)
 1. ttd - array of doubles, time profile of the shot converted to time-to-disruption values
 1. valid - boolean, whether plasma current reaches a certain value during the shot
 1. is_disruptive - boolean, 

        
For 0D data, each shot is modeled as 2D array - time vs plasma current.

## ShotList

Is a wrapper around list of shots. Therefore, it is a list of 2D arrays.

## Sublist

Shot lists is split into sublists having `num_at_once` shots from an entire dataset contained in ShotList. 

## Patch

The length of shots varies by a factor of 20. For data parallel synchronous training it is essential that amounds of train data passed to the model replica is about the same size.

Patches are subsets of shot time/signal profiles of equal length. Patch size is approximately equal to the minimum shot length (or the largest number less or equal to the minimum shot length divisible by the LSTM model length).

Since shot lengthes are not multiples of the min shot length in general, some non-deterministic fraction of patches is created.

## Chunk

A subset of `patch` defined as:
```
num_chunks = Length of the patch/ num_timesteps
```        
where `num_timesteps` is the sequence length fed to the RNN model.

## Batch

Mini-batch gradient descent is used to train neural network model.
`num_batches` represents the number of *patches* per mini-batch.

### Batch input shape

The data in batches fed to the Keras model should have shape:

```
batch_input_shape = (num_chunks*batch_size,num_timesteps,num_dimensions_of_data)
```

where `num_dimensions_of_data` is the signal dimensionality. For 0D dataset we only have a time profile of plasma current,
so `num_dimensions_of_data = 1`
