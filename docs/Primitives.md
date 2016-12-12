## Shot

Each shot is a measurement of plasma current as a function of time. The Shot objects contains following attributes:

 1. number - unique identifier of a shot (integer)
 1. t_dsirupt - disruption time in milliseconds
 1. ttd - ...
 1. valid - whether plasma current reaches a certain value during the shot

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


## Batch

