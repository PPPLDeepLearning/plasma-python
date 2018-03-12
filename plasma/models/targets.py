import numpy as np
import abc

from keras.losses import hinge, squared_hinge, mean_absolute_percentage_error
from plasma.utils.evaluation import mae_np,mse_np,binary_crossentropy_np,hinge_np,squared_hinge_np
import keras.backend as K

import plasma.conf as conf

#Requirement: larger value must mean disruption more likely.
class Target(object):
    activation = 'linear'
    loss = 'mse'

    @abc.abstractmethod
    def loss_np(y_true,y_pred):
        return conf['model']['loss_scale_factor']*mse_np(y_true,y_pred)

    @abc.abstractmethod
    def remapper(ttd,T_warning):
        return -ttd

    @abc.abstractmethod
    def threshold_range(T_warning):
        return np.logspace(-1,4,100)


class BinaryTarget(Target):
    activation = 'sigmoid'
    loss = 'binary_crossentropy'

    @staticmethod
    def loss_np(y_true,y_pred):
        return conf['model']['loss_scale_factor']*binary_crossentropy_np(y_true,y_pred)

    @staticmethod
    def remapper(ttd,T_warning,as_array_of_shots=True):
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = 0.0
        return binary_ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.logspace(-6,0,100)


class TTDTarget(Target):
    activation = 'linear'
    loss = 'mse'


    @staticmethod
    def loss_np(y_true,y_pred):
        return conf['model']['loss_scale_factor']*mse_np(y_true,y_pred)

    @staticmethod
    def remapper(ttd,T_warning):
        mask = ttd < np.log10(T_warning)
        ttd[~mask] = np.log10(T_warning)
        return -ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.linspace(-np.log10(T_warning),6,100)


class TTDInvTarget(Target):
    activation = 'linear'
    loss = 'mse'

    @staticmethod
    def loss_np(y_true,y_pred):
        return mse_np(y_true,y_pred)

    @staticmethod
    def remapper(ttd,T_warning):
        eps = 1e-4
        ttd = 10**(ttd)
        mask = ttd < T_warning
        ttd[~mask] = T_warning
        ttd = (1.0)/(ttd+eps)#T_warning
        return ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.logspace(-6,np.log10(T_warning),100)


class TTDLinearTarget(Target):
    activation = 'linear'
    loss = 'mse'

    @staticmethod
    def loss_np(y_true,y_pred):
        return conf['model']['loss_scale_factor']*mse_np(y_true,y_pred)
    

    @staticmethod
    def remapper(ttd,T_warning):
        ttd = 10**(ttd)
        mask = ttd < T_warning
        ttd[~mask] = 0#T_warning
        ttd[mask] = T_warning - ttd[mask]#T_warning
        return ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.logspace(-6,np.log10(T_warning),100)


#implements a "maximum" driven loss function. Only the maximal value in the time sequence is punished.
#Also implements class weighting
class MaxHingeTarget(Target):
    activation = 'linear'
    fac = 1.0

    @staticmethod
    def loss(y_true, y_pred):
        fac = MaxHingeTarget.fac
        #overall_fac = np.prod(np.array(K.shape(y_pred)[1:]).astype(np.float32))
        overall_fac = K.prod(K.cast(K.shape(y_pred)[1:],K.floatx()))
        max_val = K.max(y_pred,axis=-2) #temporal axis!
        max_val1 = K.repeat(max_val,K.shape(y_pred)[-2])
        mask = K.cast(K.equal(max_val1,y_pred),K.floatx())
        y_pred1 = mask * y_pred + (1-mask) * y_true
        weight_mask = K.mean(y_true,axis=-1)
        weight_mask = K.cast(K.greater(weight_mask,0.0),K.floatx()) #positive label!
        weight_mask = fac*weight_mask + (1 - weight_mask)
        #return weight_mask*squared_hinge(y_true,y_pred1)
        return conf['model']['loss_scale_factor']*overall_fac*weight_mask*hinge(y_true,y_pred1)

    @staticmethod
    def loss_np(y_true, y_pred):
        fac = MaxHingeTarget.fac
        #print(y_pred.shape)
        overall_fac = np.prod(np.array(y_pred.shape).astype(np.float32))
        max_val = np.max(y_pred,axis=-2) #temporal axis!
        max_val = np.reshape(max_val,max_val.shape[:-1] + (1,) + (max_val.shape[-1],))
        max_val = np.tile(max_val,(1,y_pred.shape[-2],1))
        mask = np.equal(max_val,y_pred)
        mask = mask.astype(np.float32)
        y_pred = mask * y_pred + (1-mask) * y_true
        weight_mask = np.greater(y_true,0.0).astype(np.float32) #positive label!
        weight_mask = fac*weight_mask + (1 - weight_mask)
        #return np.mean(weight_mask*np.square(np.maximum(1. - y_true * y_pred, 0.)))#, axis=-1) only during training, here we want to completely sum up over all instances
        return conf['model']['loss_scale_factor']*np.mean(overall_fac*weight_mask*np.maximum(1. - y_true * y_pred, 0.))#, axis=-1) only during training, here we want to completely sum up over all instances


    # def _loss_tensor_old(y_true, y_pred):
    #     max_val = K.max(y_pred) #temporal axis!
    #     mask = K.cast(K.equal(max_val,y_pred),K.floatx())
    #     y_pred = mask * y_pred + (1-mask) * y_true
    #     return squared_hinge(y_true,y_pred)


    @staticmethod
    def remapper(ttd,T_warning,as_array_of_shots=True):
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = -1.0
        return binary_ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.concatenate((np.linspace(-2,-1.06,100),np.linspace(-1.06,-0.96,100),np.linspace(-0.96,2,50)))


class HingeTarget(Target):
    activation = 'linear'

    loss = 'hinge' #hinge
    
    @staticmethod
    def loss_np(y_true, y_pred):
        return conf['model']['loss_scale_factor']*hinge_np(y_true,y_pred)
        #return squared_hinge_np(y_true,y_pred)
        
    @staticmethod
    def remapper(ttd,T_warning,as_array_of_shots=True):
        binary_ttd = 0*ttd
        mask = ttd < np.log10(T_warning)
        binary_ttd[mask] = 1.0
        binary_ttd[~mask] = -1.0
        return binary_ttd

    @staticmethod
    def threshold_range(T_warning):
        return np.concatenate((np.linspace(-2,-1.06,100),np.linspace(-1.06,-0.96,100),np.linspace(-0.96,2,50)))
