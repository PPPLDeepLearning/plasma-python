import numpy as np
epsilon = 1e-7


def get_loss_from_list(y_pred_list, y_true_list, target):
    return np.mean([get_loss(yg, yp, target)
                    for yp, yg in zip(y_pred_list, y_true_list)])


def get_loss(y_true, y_pred, target):
    return target.loss_np(y_true, y_pred)


def mae_np(y_true, y_pred):
    return np.mean(np.abs(y_pred-y_true))


def mse_np(y_true, y_pred):
    return np.mean((y_pred-y_true)**2)


def binary_crossentropy_np(y_true, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return np.mean(- (y_true*np.log(y_pred) + (1-y_true)*np.log(1 - y_pred)))


def hinge_np(y_true, y_pred):
    return np.mean(np.maximum(0.0, 1 - y_pred*y_true))


def squared_hinge_np(y_true, y_pred):
    return np.mean(np.maximum(0.0, 1 - y_pred*y_true)**2)
