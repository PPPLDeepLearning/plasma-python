from __future__ import print_function


def get_states(model):
    all_states = []
    for layer in model.layers:
        if hasattr(layer, "states"):
            layer_states = []
            for state in layer.states:
                import keras.backend as K
                layer_states.append(K.get_value(state))
            all_states.append(layer_states)
    return all_states


def set_states(model, all_states):
    i = 0
    for layer in model.layers:
        if hasattr(layer, "states"):
            layer.reset_states(all_states[i])
            i += 1


def reset_states(model, batches_to_reset):
    old_states = get_states(model)
    model.reset_states()
    new_states = get_states(model)
    for i, layer_states in enumerate(new_states):
        for j, within_layer_state in enumerate(layer_states):
            assert len(batches_to_reset) == within_layer_state.shape[0]
            within_layer_state[~batches_to_reset,
                               :] = old_states[i][j][~batches_to_reset, :]
    set_states(model, new_states)
