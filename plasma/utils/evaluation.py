import numpy as np

def get_loss_from_list(y_pred_list,y_gold_list,target):
    return np.mean([get_loss(yg,yp,target) for yp,yg in zip(y_pred_list,y_gold_list)])

def get_loss(y_gold,y_pred,target):
    return target.loss_np(y_gold,y_pred)

def mae_np(y_true,y_pred):
    return np.mean(np.abs(y_pred-y_gold))

def mse_np(y_true,y_pred):
    return np.mean((y_pred-y_gold)**2)

def binary_crossentropy_np(y_true,y_pred):
    return np.mean(- (y_gold*np.log(y_pred) + (1-y_gold)*np.log(1 - y_pred)))

def hinge_np(y_true,y_pred):
    return np.mean(np.maximum(0.0,1  - y_pred*y_gold))

def squared_hinge_np(y_true,y_pred):
    return np.mean(np.maximum(0.0,1  - y_pred*y_gold)**2)



