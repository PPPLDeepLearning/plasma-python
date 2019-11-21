'''
#########################################################
This file contains fns for printing diagnostic messages
#########################################################
'''

from __future__ import print_function
import plasma.global_vars as g


def print_shot_list_sizes(shot_list_train, shot_list_validate,
                          shot_list_test=None):
    nshots = len(shot_list_train) + len(shot_list_validate)
    nshots_disrupt = (shot_list_train.num_disruptive()
                      + shot_list_validate.num_disruptive())
    if shot_list_test is not None:
        nshots += len(shot_list_test)
        nshots_disrupt += shot_list_test.num_disruptive()
    g.print_unique('total: {} shots, {} disruptive'.format(nshots,
                                                           nshots_disrupt))
    g.print_unique('training: {} shots, {} disruptive'.format(
        len(shot_list_train), shot_list_train.num_disruptive()))
    g.print_unique('validate: {} shots, {} disruptive'.format(
        len(shot_list_validate), shot_list_validate.num_disruptive()))
    if shot_list_test is not None:
        g.print_unique('testing: {} shots, {} disruptive'.format(
            len(shot_list_test), shot_list_test.num_disruptive()))
    return
