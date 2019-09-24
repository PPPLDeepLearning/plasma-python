import numpy as np

# from keras import objectives
from keras import backend as K
from keras.losses import squared_hinge

_EPSILON = K.epsilon()


def _loss_tensor(y_true, y_pred):
    max_val = K.max(y_pred, axis=-2)  # temporal axis!
    max_val = K.repeat(max_val, K.shape(y_pred)[-2])
    print(K.eval(max_val))
    mask = K.cast(K.equal(max_val, y_pred), K.floatx())
    y_pred = mask * y_pred + (1-mask) * y_true
    return squared_hinge(y_true, y_pred)


def _loss_np(y_true, y_pred):
    print(y_pred.shape)
    max_val = np.max(y_pred, axis=-2)  # temporal axis!
    max_val = np.reshape(
        max_val, max_val.shape[:-1] + (1,) + (max_val.shape[-1],))
    max_val = np.tile(max_val, (1, y_pred.shape[-2], 1))
    print(max_val.shape)
    print(max_val)
    mask = np.equal(max_val, y_pred)
    mask = mask.astype(np.float32)
    y_pred = mask * y_pred + (1-mask) * y_true
    return np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def check_loss(_shape):
    if _shape == '2d':
        shape = (2, 3)
    elif _shape == '3d':
        shape = (2, 3, 1)
    elif _shape == '4d':
        shape = (8, 5, 6, 7)
    elif _shape == '5d':
        shape = (9, 8, 5, 6, 7)

    y_a = 1.0*np.ones(shape)
    y_b = 0.5 + np.random.random(shape)

    print(y_a)
    print(y_b)
    out1 = K.eval(_loss_tensor(K.variable(y_a), K.variable(y_b)))
    print(out1)
    out2 = _loss_np(y_a, y_b)
    print(out2)

    assert out1.shape == out2.shape
    assert out1.shape == shape[:-1]
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))


def test_loss():
    shape_list = ['3d']  # , '3d', '4d', '5d']
    for _shape in shape_list:
        check_loss(_shape)
        print('======================')


if __name__ == '__main__':
    test_loss()
