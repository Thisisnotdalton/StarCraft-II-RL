import tensorflow as tf
import numpy as np


class Constants:
    DEFAULT_ACTIONS = [
        0,  # no_op ()
        1,  # move_camera   (1/minimap [64, 64])
        5,  # select_unit   (8/select_unit_act [4]; 9/select_unit_id [500])
        7,  # select_army   (7/select_add [2])
        331,  # Move_screen (3/queued [2]; 0/screen [84, 84])
        332  # Move_minimap (3/queued [2]; 1/minimap [64, 64])
    ]

    UNIT_ELEMENTS = 7
    MAXIMUM_CARGO = 10
    MAXIMUM_BUILD_QUEUE = 10
    MAXIMUM_MULTI_SELECT = 10


def print_tensors(obj):
    for variable_name, tensor in vars(obj).items():
        if isinstance(tensor, tf.Tensor):
            print('{}:\t({} Shape={})'.format(variable_name, tensor.name, tensor.shape))


def discount_rewards(rewards, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards
