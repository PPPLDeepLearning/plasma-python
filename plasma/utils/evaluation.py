import numpy as np

def get_loss_from_list(y_pred_list,y_gold_list,mode):
    return np.mean([get_loss(yp,yg,mode) for yp,yg in zip(y_pred_list,y_gold_list)])

def get_loss(y_pred,y_gold,mode):
    if mode == 'mae':
        return np.mean(np.abs(y_pred-y_gold))
    elif mode == 'binary_crossentropy':
        return np.mean(- (y_gold*np.log(y_pred) + (1-y_gold)*np.log(1 - y_pred)))
    elif mode == 'mse':
        return np.mean((y_pred-y_gold)**2)
    elif mode == 'hinge':
        return np.mean(np.maximum(0.0,1  - y_pred*y_gold))
    elif mode == 'squared_hinge':
        return np.mean(np.maximum(0.0,1  - y_pred*y_gold)**2)
    else:
        print('mode not recognized')
        exit(1)
